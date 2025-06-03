import numpy as np
import pywt
import torch.nn as nn
import torch
from torch import nn, Tensor
import torch.nn.functional as F

# 假设 input_tensor 是一个形状为 (4, 320, 7, 7) 的 numpy 数组
input_tensor = np.random.rand(4, 320, 14, 14)

# 初始化一个空的 tensor 来存储变换后的系数
transformed_tensor = np.empty((4, 320 * 4, 7, 7), dtype=input_tensor.dtype)

# 遍历批量中的每个图像和通道，执行小波变换
for i in range(input_tensor.shape[0]):  # 遍历批量
    for c in range(input_tensor.shape[1]):  # 遍历通道
        channel_data = input_tensor[i, c, :, :]
        coeffs = pywt.dwt2(channel_data, 'haar')
        approx_coeffs, (detail_coeffs_h, detail_coeffs_v, detail_coeffs_d) = coeffs\
        # 4个数组 沿着最后一个轴（axis=-1）进行堆叠，形成一个新的多维数组。
        transformed_channel = np.stack((approx_coeffs, detail_coeffs_h, detail_coeffs_v, detail_coeffs_d), axis=-1)
        transformed_channel=transformed_channel.transpose(2,1,0)
        # 这种操作通常用于将处理后的多通道特征映射或图像数据存储在PyTorch张量中的特定位置
        transformed_tensor[i, c * 4:c * 4 + 4, :, :] = transformed_channel
        # 这种操作通常用于将处理后的多通道特征映射或图像数据存储在PyTorch张量中的特定位置。
        # 通过将transformed_channel的值复制到transformed_tensor的相应位置，
        # 可以有效地更新或填充transformed_tensor中的特定通道信息，以便进行后续的计算或处理。
    # 将numpy数组转换为Tensor张量
transformed_tensor = torch.from_numpy(transformed_tensor)

restored_tensor = np.empty((4, 320, 14, 14), dtype=input_tensor.dtype)
for i in range(transformed_tensor.shape[0]):  # 遍历批量
    for c in range(transformed_tensor.shape[1] // 4):  # 遍历通道
        transformed_channel = transformed_tensor[i, c * 4:c * 4 + 4, :, :].transpose(0,2)
        # transformed_channel = transformed_channel.transpose(2,1,0)
        approx_coeffs = transformed_channel[..., 0]
        detail_coeffs_h = transformed_channel[..., 1]
        detail_coeffs_v = transformed_channel[..., 2]
        detail_coeffs_d = transformed_channel[..., 3]

        # channel_data = pywt.idwt2((approx_coeffs, (detail_coeffs_h, detail_coeffs_v, detail_coeffs_d)), 'haar')
        channel_data = pywt.idwt2((approx_coeffs, (detail_coeffs_h, detail_coeffs_v, detail_coeffs_d)), 'haar', mode='symmetric')

        # 裁剪恢复的通道数据，使其大小与原始图像相同
        channel_data = channel_data[:14, :14]
        restored_tensor[i, c, :, :] = channel_data

restored_tensor = torch.from_numpy(restored_tensor)
#
# print(restored_tensor.shape)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, ksize=3, padding=1, bais=True):
        super(Decoder, self).__init__()



    def forward(self, x):
        # (F.interpolate(x5, scale_factor=2, mode='bilinear')

        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

if __name__ == '__main__':
    model = BasicConv2d(320,320,3,1,1).cuda()
    x = torch.randn(4,320,14,14).cuda()
    x = model(x).cuda()
    print(x.shape)