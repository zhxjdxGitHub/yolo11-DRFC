import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ultralytics.nn.Addmodules.visualize_4d_tensor import *
from ultralytics.nn.modules.conv import autopad, Conv
import cv2
import torch
import numpy as np
import torch.nn.functional  as F
import math
from ultralytics.nn.Addmodules.TensorRotator4D import *
import os
os.environ['DISPLAY']  = ':0'  # 指定显示设备

from pycuda import driver as drv
from pycuda.compiler import SourceModule
num=0


__all__ = ['Fuzzy_Conv']

import torch


def rotate_4d_tensor(tensor, angle_degrees=45):
    """
    对四维张量（N, C, H, W）旋转指定角度（默认45度）
    参数：
        tensor: 输入张量，形状如 (1, 256, 20, 20)
        angle_degrees: 旋转角度（顺时针为正）
    返回：
        旋转后的张量（形状不变，边界可能填充0）
    """
    if tensor.dim() != 4:
        raise ValueError("输入必须是四维张量 (N, C, H, W)")

    N, C, H, W = tensor.shape
    angle_rad = math.radians(angle_degrees)

    # 构造旋转矩阵（2x3），数据类型与输入张量一致
    rotation_matrix = torch.tensor([
        [math.cos(angle_rad), -math.sin(angle_rad), 0],
        [math.sin(angle_rad), math.cos(angle_rad), 0]
    ], dtype=tensor.dtype, device=tensor.device)  # 自动匹配dtype

    # 扩展旋转矩阵到批量大小 (N, 2, 3)
    rotation_matrix = rotation_matrix.unsqueeze(0).repeat(N, 1, 1)

    # 生成采样网格（数据类型自动继承）
    grid = F.affine_grid(rotation_matrix, tensor.size(), align_corners=False)

    # 执行旋转（自动处理数据类型）
    rotated_tensor = F.grid_sample(
        tensor,
        grid,
        align_corners=False,
        mode='bilinear',
        padding_mode='zeros'
    )

    return rotated_tensor

def safe_device_merge(switch, avg_x, numpy_result):
    """安全设备合并（自动处理CPU/GPU混合）"""
    # 统一设备（2025年新增DDP兼容逻辑）
    target_device = avg_x.device if torch.is_tensor(avg_x) else \
        torch.device(f'cuda:{torch.cuda.current_device()}')

    # 强制转换numpy结果到目标设备
    if isinstance(numpy_result, (np.ndarray, np.generic)):
        tensor_result = torch.from_numpy(numpy_result).to(target_device)
    else:
        tensor_result = numpy_result.to(target_device)

        # 类型检查（2025年新增AMP混合精度支持）
    if switch.dtype != tensor_result.dtype:
        tensor_result = tensor_result.type(switch.dtype)

    return switch * avg_x + (1 - switch) * tensor_result

import cv2
def safe_color_convert(img):
    # 通道数检查与修复
    if img.ndim  == 2:  # 灰度图
        img = cv2.cvtColor(img,  cv2.COLOR_GRAY2BGR)
    elif img.shape[-1]  == 4:  # RGBA
        img = cv2.cvtColor(img,  cv2.COLOR_RGBA2BGR)
    elif img.shape[-1]  == 3:  # RGB
        img = cv2.cvtColor(img,  cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"非法通道数：{img.shape[-1]} ，仅支持1/3/4通道")
    return img

def show_image_cv(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"图片路径 '{image_path}' 不存在或格式不支持")

        cv2.imshow("OpenCV  Window", image)
        cv2.waitKey(0)  # 按任意键关闭窗口
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"错误: {e}")



class Fuzzy_Conv(nn.Conv2d):
    """左平移输入张量n个像素后与原始输入相加的模块"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 s=1,
                 p=None,
                 g=1,
                 d=1,
                 act=True,
                 bias=True,
                 n_pixels=40):
        """
        Args:
            n_pixels (int): 左平移的像素数，默认为1
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=s,
            padding=p,
            groups=g,
            bias=bias)
        self.n_pixels = n_pixels

        self.pre_context = torch.nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)

        self.switch = torch.nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=1,
            stride=s,
            bias=True)
        self.switch.weight.data.fill_(0)
        self.switch.bias.data.fill_(1)

        self.post_context = torch.nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=1,
            bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        global num  # 函数内声明使用全局变量 
        # pre-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x
        # 可视化并保存
        saved_path = visualize_4d_tensor(x,num,'./vis_results(1)')
        # print(f"最终保存路径（1）: {saved_path}")

        # switch
        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)

        #旋转
        output_tensor = rotate_4d_tensor(avg_x, angle_degrees=10)

        # 可视化并保存
        saved_path = visualize_4d_tensor(output_tensor,num,'./vis_results(2)')
        # print(f"最终保存路径（2）: {saved_path}")

        #拼接
        out = switch * avg_x + (1 - switch) * output_tensor

        #后期处理
        avg_x = torch.nn.functional.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x

        out= self.act(self.bn(out))
        # 可视化并保存
        saved_path = visualize_4d_tensor(out,num,'./vis_results(3)')
        # print(f"最终保存路径（3）: {saved_path}")
        # num+=1
        return out

if __name__ == "__main__":
    # 初始化模块
    shift_add = Fuzzy_Conv(n_pixels=20)

    # 读取并预处理图像
    image = cv2.imread("/mnt/3C6EF7006EF6B22E/nzh/mypro/snu-YOLOv11/1.png")
    assert image is not None, "图片读取失败，请检查路径"

    # 转换为RGB并归一化到[0,1]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    x = torch.from_numpy(image_rgb).permute(2, 0, 1).float()  # [H,W,C] -> [C,H,W]

    # 处理并保存结果
    output = shift_add(x)
    output_np = output.permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C]

    # 确保数值范围正确并转换颜色空间
    output_np = (np.clip(output_np, 0, 1) * 255).astype('uint8')
    cv2.imwrite('/mnt/3C6EF7006EF6B22E/nzh/mypro/snu-YOLOv11/output.png', cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR))
    # 使用示例
    # show_image_cv("/mnt/3C6EF7006EF6B22E/nzh/mypro/snu-YOLOv11/1.png")
    # plt.savefig("/mnt/3C6EF7006EF6B22E/nzh/mypro/snu-YOLOv11/output.png", bbox_inches='tight', dpi=300)

    print(output)