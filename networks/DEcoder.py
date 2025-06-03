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
        y_spatial = self.spatial_attn(x)           #32,1,256,128
        y_channel = self.channel_attn(x)           #32,16,1,1
        y = y_spatial * y_channel                  #32,16,256,128
        x = torch.sigmoid(self.conv(y))
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


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



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

class lws(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(lws, self).__init__()
        self.wave = WaveletConv2d(in_channel, out_channel, wavelet='haar')
    def forward(self, x):
        x = self.wave(x)

        return x

class upconv(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(upconv, self).__init__()
        self.conv = BasicConv2d(in_channel, in_channel, kernel_size=1)
        self.sa = SoftAttn(in_channel, in_channel)
        self.conv3 = BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel)

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

class waveletBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(waveletBlock, self).__init__()
        self.bn = nn.BatchNorm2d(out_channel)
        self.lws = lws(in_channel, out_channel)
        self.ca = ChannelAttention(in_channel)
        self.conv = BasicConv2d(in_channel, out_channel,kernel_size=1)
        self.conv3 = BasicConv2d(in_channel, out_channel, kernel_size=3,stride=1,padding=1)

    def forward(self, x, x1):
        x = x + x1
        x_replica1 = self.lws(self.bn(x))
        x_replica2 = self.lws(x_replica1)
        x_replica2 = self.conv(x_replica2 * self.ca(x_replica1))
        x = x_replica2 + x
        x_replica3 = self.conv(self.bn(x))
        x_replica3 = self.conv(x_replica3 * torch.sigmoid(self.conv(self.bn(x))))
        x = x + x_replica3
        x1 = self.conv3(x)
        return x1


class waveletBlock_5(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(waveletBlock_5, self).__init__()
        self.bn = nn.BatchNorm2d(out_channel)
        self.lws = lws(in_channel, out_channel)
        self.ca = ChannelAttention(in_channel)
        self.conv = BasicConv2d(in_channel, out_channel, 1)
        self.conv3 = BasicConv2d(in_channel, out_channel, 3,1,1)

    def forward(self, x):
        x_replica1 = self.lws(self.bn(x))
        x_replica2 = self.lws(x_replica1)
        x_replica2 = self.conv(x_replica2 * self.ca(x_replica1))
        x = x_replica2 + x
        x_replica3 = self.conv(self.bn(x))
        x_replica3 = self.conv(x_replica3 * torch.sigmoid(self.conv(self.bn(x))))
        x =self.conv3(x + x_replica3)
        return x
class Decoder(nn.Module):
    def __init__(self,  embed_dims=[64, 128, 320, 512]):
        super(Decoder, self).__init__()
        self.dims = embed_dims
        self.upconv3 = upconv(embed_dims[3], embed_dims[2])
        self.upconv2 = upconv(embed_dims[2], embed_dims[1])
        self.upconv1 = upconv(embed_dims[1], embed_dims[0])
        self.waveletBlock_4 = waveletBlock_5(embed_dims[3], embed_dims[3])
        self.waveletBlock_3 = waveletBlock(embed_dims[2], embed_dims[2])
        self.waveletBlock_2 = waveletBlock(embed_dims[1], embed_dims[1])
        self.waveletBlock_1 = waveletBlock(embed_dims[0], embed_dims[0])
        self.conv3 = BasicConv2d(embed_dims[2], embed_dims[0])
        self.conv2 = BasicConv2d(embed_dims[1], embed_dims[0])
        # self.lws = lws(3)
    def forward(self, x):
        x[3] = self.waveletBlock_4(x[3])
        x[3] = self.upconv3(x[3])             # x[3] = (4, 320, 14, 14)
        x[2] = self.waveletBlock_3(x[2], x[3])
        x[2] = self.upconv2(x[2])             # x[2] = (4, 128, 28, 28)
        x[1] = self.waveletBlock_2(x[1], x[2])
        x[1] = self.upconv1(x[1])             # x[1] = (4, 64, 56, 56)
        x[0] = self.waveletBlock_1(x[0], x[1])
        x1 = x[0]
        return x1




if __name__ == '__main__':

    p1 = torch.randn(4, 64, 56, 56).cuda()
    p2 = torch.randn(4, 128, 28, 28).cuda()
    p3 = torch.randn(4, 320, 14, 14).cuda()
    p4 = torch.randn(4, 512, 7, 7).cuda()
    x = [p1, p2, p3, p4]
    model = Decoder().cuda()
    out = model(x)
    print(x[1].shape)
    print(x[2].shape)
    print(x[3].shape)
    print(x[0].shape)
    print(out.shape)