import numpy
import torch
from torch import nn, Tensor
from .UEncoder import UEncoder
import torch.nn.functional as F


class SideConv(nn.Module):
    def __init__(self, n_classes=9):
        super(SideConv, self).__init__()

        self.side5 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.side4 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.side3 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.side2 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.side1 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, stage_feat):
        x5, x5_up, x6_up, x7_up, x8_up = stage_feat[0], stage_feat[1], stage_feat[2], stage_feat[3], stage_feat[4]
        out5 = self.side5(x5)
        out5 = self.upsamplex2(out5)
        out5 = self.upsamplex2(out5)
        out5 = self.upsamplex2(out5)
        out5 = self.upsamplex2(out5)

        out4 = self.side4(x5_up)
        out4 = self.upsamplex2(out4)
        out4 = self.upsamplex2(out4)
        out4 = self.upsamplex2(out4)

        out3 = self.side3(x6_up)
        out3 = self.upsamplex2(out3)
        out3 = self.upsamplex2(out3)

        out2 = self.side2(x7_up)
        out2 = self.upsamplex2(out2)

        out1 = self.side1(x8_up)
        return [out5, out4, out3, out2, out1]

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
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
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.u_encoder = UEncoder()

        self.rfb1 = RFB_modified(64, 64)
        self.rfb2 = RFB_modified(128, 64)
        self.rfb3 = RFB_modified(256, 64)
        self.rfb4 = RFB_modified(512, 64)
        self.rfb5 = RFB_modified(1024, 64)

    def forward(self, x, en=[]):

        encoder_features = self.u_encoder(x)
        e1 = self.rfb1(encoder_features[0])  # torch.Size([4, 64, 224, 224])
        e2 = self.rfb2(encoder_features[1])  # torch.Size([4, 64, 112, 112])
        e3 = self.rfb3(encoder_features[2])  # torch.Size([4, 64, 56, 56])
        e4 = self.rfb4(encoder_features[3])  # torch.Size([4, 64, 28, 28])
        e5 = self.rfb5(encoder_features[4])  # torch.Size([4, 64, 14, 14])

        if len(en) != 0:
            ''' 
            1、是否需要加入平滑卷积
            2、差分非空，加入差分
            '''
            e1 = e1 + en[0]
            e2 = e2 + en[1]
            e3 = e3 + en[2]
            e4 = e4 + en[3]
            e5 = e5 + en[4]

        out = [e1, e2, e3, e4, e5]

        return out


class UNetDecoder(nn.Module):
    def __init__(self, n_classes=9):
        super(UNetDecoder, self).__init__()

        self.x5_4 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.x4_3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.x3_2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.x2_1 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.x5_4_3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.x4_3_2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.x3_2_1 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.x5_4_3_2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.x4_3_2_1 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.x5_4_3_2_1 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        # if 平滑卷积
        self.level3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.level2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.level1 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        # else 平滑卷积
        self.smooth1 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.output4 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.output3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.output2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.output1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1))

        self.pred = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, features, f1='none'):

        x1 = features[0]  # torch.Size([4, 64, 224, 224])
        x2 = features[1]  # torch.Size([4, 64, 112, 112])
        x3 = features[2]  # torch.Size([4, 64, 56, 56])
        x4 = features[3]  # torch.Size([4, 64, 28, 28])
        x5 = features[4]  # torch.Size([4, 64, 14, 14])

        if f1 == 'none':
            '''
                1、是否对RFB加入平滑卷积
            '''
            x5_4 = self.x5_4(abs(F.interpolate(x5, scale_factor=2, mode='bilinear') - x4))  # torch.Size([4, 64, 28, 28])
            x4_3 = self.x4_3(abs(F.interpolate(x4, scale_factor=2, mode='bilinear') - x3))  # torch.Size([4, 64, 56, 56])
            x3_2 = self.x3_2(abs(F.interpolate(x3, scale_factor=2, mode='bilinear') - x2))  # torch.Size([4, 64, 112, 112])
            x2_1 = self.x2_1(abs(F.interpolate(x2, scale_factor=2, mode='bilinear') - x1))  # torch.Size([4, 64, 224, 224])

            x5_4_3 = self.x5_4_3(abs(F.interpolate(x5_4, scale_factor=2, mode='bilinear') - x4_3))  # torch.Size([4, 64, 56, 56])
            x4_3_2 = self.x4_3_2(abs(F.interpolate(x4_3, scale_factor=2, mode='bilinear') - x3_2))  # torch.Size([4, 64, 112, 112])
            x3_2_1 = self.x3_2_1(abs(F.interpolate(x3_2, scale_factor=2, mode='bilinear') - x2_1))  # torch.Size([4, 64, 224, 224])

            x5_4_3_2 = self.x5_4_3_2(abs(F.interpolate(x5_4_3, scale_factor=2, mode='bilinear') - x4_3_2))  # torch.Size([4, 64, 112, 112])
            x4_3_2_1 = self.x4_3_2_1(abs(F.interpolate(x4_3_2, scale_factor=2, mode='bilinear') - x3_2_1))  # torch.Size([4, 64, 224, 224])

            x5_4_3_2_1 = self.x5_4_3_2_1(abs(F.interpolate(x5_4_3_2, scale_factor=2, mode='bilinear') - x4_3_2_1))  # torch.Size([4, 64, 224, 224])

            level4 = x5_4                                                 # torch.Size([4, 64, 28, 28]) 差分特征
            level3 = self.level3(x4_3 + x5_4_3)                           # torch.Size([4, 64, 56, 56]) 差分特征
            level2 = self.level2(x3_2 + x4_3_2 + x5_4_3_2)                # torch.Size([4, 64, 112, 112]) 差分特征
            level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)   # torch.Size([4, 64, 224, 224]) 差分特征

            output5 = x5                                                                              # torch.Size([4, 64, 14, 14])
            output4 = self.output4(F.interpolate(x5, scale_factor=2, mode='bilinear') + level4)       # torch.Size([4, 64, 28, 28])
            output3 = self.output3(F.interpolate(output4, scale_factor=2, mode='bilinear') + level3)  # torch.Size([4, 64, 56, 56])
            output2 = self.output2(F.interpolate(output3, scale_factor=2, mode='bilinear') + level2)  # torch.Size([4, 64, 112, 112])
            output1 = self.output1(F.interpolate(output2, scale_factor=2, mode='bilinear') + level1)  # torch.Size([4, 64, 224, 224])

            output = self.pred(output1)

        else:
            '''
                1、f1:解码器1
                2、w5, w4, w3, w2, w1将不再具有梯度信息
            '''
            m5, m4, m3, m2, m1 = f1[4], f1[3], f1[2], f1[1], f1[0]
            w5, w4, w3, w2, w1 = torch.sigmoid(m5), torch.sigmoid(m4), torch.sigmoid(m3), torch.sigmoid(m2), torch.sigmoid(m1)
            x5_s = self.smooth1(x5 + x5 * w5)      # torch.Size([4, 64, 14, 14])

            x5_4 = self.x5_4(abs(F.interpolate(x5_s, scale_factor=2, mode='bilinear') + x4 * (1 - w4)))  # torch.Size([4, 64, 28, 28])
            x4_3 = self.x4_3(abs(F.interpolate(x4, scale_factor=2, mode='bilinear') + x3 * (1 - w3)))    # torch.Size([4, 64, 56, 56])
            x3_2 = self.x3_2(abs(F.interpolate(x3, scale_factor=2, mode='bilinear') + x2 * (1 - w2)))    # torch.Size([4, 64, 112, 112])
            x2_1 = self.x2_1(abs(F.interpolate(x2, scale_factor=2, mode='bilinear') + x1 * (1 - w1)))    # torch.Size([4, 64, 224, 224])

            x5_4_3 = self.x5_4_3(abs(F.interpolate(x5_4, scale_factor=2, mode='bilinear') - x4_3))  # torch.Size([4, 64, 56, 56])
            x4_3_2 = self.x4_3_2(abs(F.interpolate(x4_3, scale_factor=2, mode='bilinear') - x3_2))  # torch.Size([4, 64, 112, 112])
            x3_2_1 = self.x3_2_1(abs(F.interpolate(x3_2, scale_factor=2, mode='bilinear') - x2_1))  # torch.Size([4, 64, 224, 224])

            x5_4_3_2 = self.x5_4_3_2(abs(F.interpolate(x5_4_3, scale_factor=2, mode='bilinear') - x4_3_2))  # torch.Size([4, 64, 112, 112])
            x4_3_2_1 = self.x4_3_2_1(abs(F.interpolate(x4_3_2, scale_factor=2, mode='bilinear') - x3_2_1))  # torch.Size([4, 64, 224, 224])

            x5_4_3_2_1 = self.x5_4_3_2_1(abs(F.interpolate(x5_4_3_2, scale_factor=2, mode='bilinear') - x4_3_2_1))  # torch.Size([4, 64, 224, 224])

            level4 = x5_4                                                 # torch.Size([4, 64, 28, 28])
            level3 = self.level3(x4_3 + x5_4_3)                           # torch.Size([4, 64, 56, 56])
            level2 = self.level2(x3_2 + x4_3_2 + x5_4_3_2)                # torch.Size([4, 64, 112, 112])
            level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)   # torch.Size([4, 64, 224, 224])

            output5 = x5_s
            output4 = self.output4(F.interpolate(x5_s, scale_factor=2, mode='bilinear') + level4)     # torch.Size([4, 64, 28, 28])
            output3 = self.output3(F.interpolate(output4, scale_factor=2, mode='bilinear') + level3)  # torch.Size([4, 64, 56, 56])
            output2 = self.output2(F.interpolate(output3, scale_factor=2, mode='bilinear') + level2)  # torch.Size([4, 64, 112, 112])
            output1 = self.output1(F.interpolate(output2, scale_factor=2, mode='bilinear') + level1)  # torch.Size([4, 64, 224, 224])

            output = self.pred(output1)

        return output, [output1, output2, output3, output4, output5]



class MGRAD-UNet(nn.Module):
    def __init__(self, n_classes=9):
        super(MGRAD-UNet, self).__init__()
        self.n_classes = n_classes
        self.encoder = Encoder()
        self.decoder1 = UNetDecoder(n_classes)
        self.decoder2 = UNetDecoder(n_classes)
        self.side_conv = SideConv(n_classes)

    def forward(self, input, en=[]):
        if input.size()[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        features = self.encoder(input, en)
        out_seg1, stage_feat1 = self.decoder1(features)
        out_seg2, stage_feat2 = self.decoder2(features, stage_feat1)
        rev_stage_feat1 = []
        for i in range(len(stage_feat1)):
            rev_stage_feat1.append(stage_feat1[4-i])
        deep_out1 = self.side_conv(rev_stage_feat1)
        return out_seg1, out_seg2, [stage_feat2, stage_feat1], deep_out1, []


if __name__ == '__main__':
    model = MGRAD-UNet().cuda()
    x = torch.randn(4, 3, 224, 224).cuda()
    for num in range(3):
        if num == 0:
            outputs1, outputs2, masks, stage_out1, _ = model(x, [])
            print(stage_out1[0].shape)
            print(stage_out1[1].shape)
            print(stage_out1[2].shape)
            print(stage_out1[3].shape)
            print(stage_out1[4].shape)
        else:
            outputs1, outputs2, masks, stage_out1, _ = model(x, en)
            print(stage_out1[0].shape)
            print(stage_out1[1].shape)
            print(stage_out1[2].shape)
            print(stage_out1[3].shape)
            print(stage_out1[4].shape)
        en = []
        for idx in range(len(masks[0])):
            mask1 = masks[0][idx].detach()
            mask2 = masks[1][idx].detach()
            en.append(1e-3 * (mask1 - mask2))
        out5, out4, out3, out2, out1 = stage_out1[0], stage_out1[1], stage_out1[2], stage_out1[3], stage_out1[4]
        out1_soft = F.softmax(out1, dim=1)
        out2_soft = F.softmax(out2, dim=1)
        out3_soft = F.softmax(out3, dim=1)
        out4_soft = F.softmax(out4, dim=1)
        out5_soft = F.softmax(out5, dim=1)

        outputs_soft1 = F.softmax(outputs1, dim=1)
        outputs_soft2 = F.softmax(outputs2, dim=1)





