import numpy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from waveletConv import WaveletConv2d

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



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.sigmoid(out)
class SoftAttn(nn.Module): #软注意力（32，16，256，128）=空间注意力的输出（32，1，256，128）乘上通道注意力（32,16,1,1）

    def __init__(self, in_channel, out_channel):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttention()
        self.channel_attn = ChannelAttention(in_channel)
        self.conv = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):#x.shape(32,16,256,128)
        x_spatial = self.spatial_attn(x)           #32,1,256,128
        x_channel = self.channel_attn(x)           #32,16,1,1
        x = x_spatial * x_channel                  #32,16,256,128
        x = torch.sigmoid(self.conv(x))
        return x                             #torch.Size([32, 16, 256, 128])


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return torch.sigmoid(x)

class upconv(nn.Module):
    def __init__(self, in_channel,out_channel=None):
        super(upconv, self).__init__()
        self.conv = BasicConv2d(in_channel, in_channel, kernel_size=1)
        self.sa = SoftAttn(in_channel, in_channel)
        self.conv3 = BasicConv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        x = self.conv(x)
        x_sa = self.sa(x)
        # 使用torch.cat进行拼接，需要确保x和x*x_sa的维度匹配
        x = x * x_sa
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
if __name__ == '__main__':
    model = upconv(320).cuda()
    x = torch.randn(4,320,14,14).cuda()
    x = model(x).cuda()
    print(x.shape)


