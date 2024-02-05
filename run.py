import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

#from scipy.optimize import curve_fit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--dep-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    
    args = parser.parse_args()
    
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(DEVICE)
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    if os.path.isfile(args.img_path): #自动将img-path的-替换为了_
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = os.listdir(args.img_path) #列出目录下所有文件名
        filenames = [os.path.join(args.img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()

    if os.path.isfile(args.dep_path): #自动将dep-path的-替换为了_
        if args.dep_path.endswith('txt'):
            with open(args.dep_path, 'r') as f:
                dep_filenames = f.read().splitlines()
        else:
            dep_filenames = [args.dep_path]
    else:
        dep_filenames = os.listdir(args.dep_path)
        dep_filenames = [os.path.join(args.dep_path, dep_filename) for dep_filename in dep_filenames if not dep_filename.startswith('.')]
        dep_filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    for filename, dep_filename in tqdm(zip(filenames, dep_filenames)):
        np.set_printoptions(threshold=np.inf)
        raw_image = cv2.imread(filename)
        rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0 # 归一化，为了输入神经网络
        h, w = rgb_image.shape[:2]
        rgb_trans = transform({'image': rgb_image})['image'] # shape(518,686) 预处理，为了输入神经网络
        rgb_trans = torch.from_numpy(rgb_trans).unsqueeze(0).to(DEVICE) # numpy->tensor，为了输入神经网络

        dep_image = cv2.imread(dep_filename, cv2.IMREAD_UNCHANGED)
        dep_raw = np.copy(dep_image)
        # tum/bonn rgbd数据集相机的depth factor = 5000.0，除以该值后，像素值为真实浮点距离；realsense 应改为 1000.0
        dep_image = dep_image / 5000.0
        
        with torch.no_grad():
            depth_predict = depth_anything(rgb_trans) # 输出预测深度图
        depth_predict = F.interpolate(depth_predict[None], (h, w), mode='bilinear', align_corners=False)[0, 0] # shape(480,640) ，插值恢复原分辨率
        depth_predict = depth_predict.cpu().numpy() # tensor->numpy，恢复numpy类型
        # depth_valid1 =  dep_image > 0                        # 布尔有效索引矩阵，用于索引获取N个有效深度值，只要有深度值就用
        depth_valid2 = (dep_image > 0.5) & (dep_image < 3.0) # 布尔有效索引矩阵，用于索引获取N个有效深度值，只获取精度值更高的深度值

        # b = Ax -> A = bx^{-1}
        # sensor深度图像有效值b = A * predict深度图像有效值x，A=[scale,offset]，b = A[0] * x[0] + A[1] * x[1] = A[0] * x[0] + A[1]，其中 x[1]≡1
        x = np.stack(
            (depth_predict[depth_valid2], np.ones_like(depth_predict[depth_valid2])), axis=1
        ).T
        b = dep_image[depth_valid2].T # shape: dep_image[depth_valid]=[N,1]
        pinvx = np.linalg.pinv(x)
        A = b @ pinvx                 # shape: 矩阵乘法 A = [1,N] @ [N,2] 
        depth_adjusted_scale = depth_predict * A[0] + A[1]

        # # 另一种调库方法：定义线性模型
        # def linear_model(x, a, b):
        #     return a * x + b
        # # 将深度图像有效值组合成[x, 1]形式的矩阵
        # x_data = depth_predict[depth_valid2]
        # y_data = dep_image[depth_valid2]
        # # 使用curve_fit进行线性模型的拟合
        # params, covariance = curve_fit(linear_model, x_data, y_data)
        # # 获取拟合参数
        # A2 = params
        
        invalid_points = dep_image == 0
        dep_image[invalid_points] = depth_adjusted_scale[invalid_points] # 使用条件索引，将dep_image中的无效点替换为adjusted对应位置的值


        # 打印参数
        # print("x.shape",x.shape) # shape: depth_predict[depth_valid]=[N,1]   stack=[N,2], x=stack.T=[2,N]  N个有效值
        # print("b.shape",b.shape) # shape: b[1,N]
        # print("pinvx.shape",pinvx.shape) # shape: pinvx[N,2]
        # print("A",A)                     # shape: A[1,2]  
        # # print("A2",A2)          
        
        RMSE = np.sqrt(((A @ x - b) ** 2).mean())
        REL = (np.abs(A @ x - b) / b).mean()
        print("RMSE",RMSE)
        print("REL",REL)

        # visualizaiton
        cv2.imshow("raw image",raw_image)

        # dep_raw = (dep_raw - dep_raw.min()) / (dep_raw.max() - dep_raw.min()) * 255.0
        dep_raw = (dep_raw / 5000.0 ) * 60.0 #为了更好的查看真实深度值，将深度值范围（0~3.0+）映射为（0~180.0+）
        dep_raw = dep_raw.astype(np.uint8)
        cv2.imshow("raw depth",dep_raw)

        depth_predict = (depth_predict - depth_predict.min()) / (depth_predict.max() - depth_predict.min()) * 255.0
        depth_predict = depth_predict.astype(np.uint8)
        cv2.imshow("AI predict depth",depth_predict)

        # depth_adjusted_scale_ = (depth_adjusted_scale - depth_adjusted_scale.min()) / (depth_adjusted_scale.max() - depth_adjusted_scale.min()) * 255.0
        depth_adjusted_scale_ = depth_adjusted_scale * 60.0
        depth_adjusted_scale_ = depth_adjusted_scale_.astype(np.uint8)
        cv2.imshow("recovery scale depth",depth_adjusted_scale_)
        # print("depth_adjusted_scale min ",depth_adjusted_scale.min())
        # print("depth_adjusted_scale max ",depth_adjusted_scale.max())

        # dep_fillhole = (dep_image - dep_image.min()) / (dep_image.max() - dep_image.min()) * 255.0
        dep_fillhole = dep_image * 60.0
        dep_fillhole = dep_fillhole.astype(np.uint8)
        cv2.imshow("raw depth only fill holes",dep_fillhole)
        # print("dep_fillhole min ",dep_image.min())
        # print("dep_fillhole max ",dep_image.max())

        cv2.waitKey(0)
