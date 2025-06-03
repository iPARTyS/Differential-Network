import torch
import torch.nn as nn
import pywt
import numpy as np
import numba.cuda as cuda


class DWconv(nn.Module):
    def __init__(self, in_channel, out_channel, ksize=3, padding=1, bais=True):
        super(DWconv, self).__init__()

        self.depthwiseConv = nn.Conv2d(in_channels=in_channel,
                                       out_channels=out_channel,
                                       groups=in_channel,
                                       kernel_size=ksize,
                                       padding=padding,
                                       bias=bais)
        self.pointwiseConv = nn.Conv2d(in_channels=in_channel,
                                       out_channels=out_channel,
                                       kernel_size=1,
                                       padding=0,
                                       bias=bais)

    def forward(self, x):
        out = self.depthwiseConv(x)
        out = self.pointwiseConv(out)
        return out

class WaveletConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, wavelet='haar'):
        super(WaveletConv2d, self).__init__()
        self.in_channels = 4 * in_channels
        self.out_channels = 4 * out_channels
        self.wavelet = wavelet
        self.dwconv = DWconv(self.in_channels, self.out_channels)
        # 初始化小波滤波器
        # self.wavelet_filter = torch.Tensor(pywt.Wavelet(self.wavelet).dec_lo)

        # 定义卷积层
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
    def forward(self, x):
        # 小波变换
        x_np = x.cpu().detach().numpy()
        B, C, H, W = x_np.shape
        if H%2 != 0 :
            x_tensor = np.empty((B, 4 * C, (H+1) // 2, (W+1) // 2), dtype=x_np.dtype)
        else:
            x_tensor = np.empty((B, 4*C, H//2, W//2), dtype=x_np.dtype)

        for i in range(x_np.shape[0]):  # 遍历批量
            for c in range(x_np.shape[1]):  # 遍历通道
                channel_data = x_np[i, c, :, :]
                coeffs = pywt.dwt2(channel_data, 'haar')
                approx_coeffs, (detail_coeffs_h, detail_coeffs_v, detail_coeffs_d) = coeffs \
                    # 4个数组 沿着最后一个轴（axis=-1）进行堆叠，形成一个新的多维数组。
                transformed_channel = np.stack((approx_coeffs, detail_coeffs_h, detail_coeffs_v, detail_coeffs_d), axis=-1)
                transformed_channel = transformed_channel.transpose(2, 1, 0)
                x_tensor[i, c * 4:c * 4 + 4, :, :] = transformed_channel

        transformed_tensor = torch.from_numpy(x_tensor).cuda()
        transformed_tensor = self.dwconv(transformed_tensor)
        transformed_tensor = self.conv(transformed_tensor)
        transformed_tensor = self.gelu(transformed_tensor)
        transformed_tensor = transformed_tensor.cpu()
        restored_tensor = np.empty((B, C, H, W), dtype=x_np.dtype)
        if H % 2 != 0 :
            for i in range(transformed_tensor.shape[0]):  # 遍历批量
                for c in range(transformed_tensor.shape[1] // 4):  # 遍历通道
                    transformed_channel = transformed_tensor[i, c * 4:c * 4 + 4, :, :]
                    approx_coeffs = transformed_channel[..., 0].detach().numpy()
                    detail_coeffs_h = transformed_channel[..., 1].detach().numpy()
                    detail_coeffs_v = transformed_channel[..., 2].detach().numpy()
                    detail_coeffs_d = transformed_channel[..., 3].detach().numpy()
                    channel_data = pywt.idwt2((approx_coeffs, (detail_coeffs_h, detail_coeffs_v, detail_coeffs_d)), 'haar', mode='symmetric')
                    # 裁剪恢复的通道数据，使其大小与原始图像相同
                    channel_data = channel_data[:H, :W]
                    restored_tensor[i, c, :, :] = channel_data
        else:
            for i in range(transformed_tensor.shape[0]):  # 遍历批量
                for c in range(transformed_tensor.shape[1] // 4):  # 遍历通道
                    transformed_channel = transformed_tensor[i, c * 4:c * 4 + 4, :, :].transpose(0,2)
                    approx_coeffs = transformed_channel[..., 0].detach().numpy()
                    detail_coeffs_h = transformed_channel[..., 1].detach().numpy()
                    detail_coeffs_v = transformed_channel[..., 2].detach().numpy()
                    detail_coeffs_d = transformed_channel[..., 3].detach().numpy()
                    channel_data = pywt.idwt2((approx_coeffs, (detail_coeffs_h, detail_coeffs_v, detail_coeffs_d)), 'haar')
                    restored_tensor[i, c, :, :] = channel_data
        restored_tensor = torch.from_numpy(restored_tensor).cuda()
        x = restored_tensor
        return x




if __name__ == '__main__':

    p1 = torch.randn(4, 64, 56, 56).cuda()
    p2 = torch.randn(4, 128, 28, 28).cuda()
    p3 = torch.randn(4, 320, 14, 14).cuda()
    p4 = torch.randn(4, 512, 7, 7).cuda()
    model = WaveletConv2d(512,512).cuda()
    out = model(p4)
    print(out.shape)

