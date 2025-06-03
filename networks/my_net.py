import numpy
import torch
from torch import nn, Tensor
from .Unet import up_conv, conv_block
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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
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
class UNetEncoder(nn.Module):
    def __init__(self, in_ch=3):
        super(UNetEncoder, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]   # 64, 128, 256, 512, 1024

        # 最大池化层
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.rfb2_1 = RFB_modified(64, 128)
        self.rfb3_1 = RFB_modified(64, 256)
        self.rfb4_1 = RFB_modified(64, 512)
        self.rfb5_1 = RFB_modified(64, 1024)

        # 左边特征提取卷积层
        self.Conv1 = conv_block(in_ch, filters[0])       # 3 -> 64
        self.Conv2 = conv_block(filters[0], filters[1])  # 64 -> 128
        self.Conv3 = conv_block(filters[1], filters[2])  # 128 -> 256
        self.Conv4 = conv_block(filters[2], filters[3])  # 256 -> 512
        self.Conv5 = conv_block(filters[3], filters[4])  # 512 -> 1024

    def forward(self, x , en=[]):
        if len(en) == 0:
            e1 = self.Conv1(x)               # torch.size([1, 64, 224, 224])

            e2 = self.Maxpool1(e1)           # torch.size([1, 64, 112, 112])
            e2 = self.Conv2(e2)              # torch.size([1, 128, 112, 112])

            e3 = self.Maxpool2(e2)           # torch.size([1, 128, 56, 56])
            e3 = self.Conv3(e3)              # torch.size([1, 256, 56, 56])

            e4 = self.Maxpool3(e3)           # torch.size([1, 256, 28, 28])
            e4 = self.Conv4(e4)              # torch.size([1, 512, 28, 28])

            e5 = self.Maxpool4(e4)           # torch.size([1, 512, 14, 14])
            e5 = self.Conv5(e5)              # torch.size([1, 1024, 14, 14])

        else:
            e1 = self.Conv1(x) + en[0]

            e2 = self.Maxpool1(e1)
            e2 = self.Conv2(e2) + self.rfb2_1(en[1])

            e3 = self.Maxpool2(e2)
            e3 = self.Conv3(e3) + self.rfb3_1(en[2])

            e4 = self.Maxpool3(e3)
            e4 = self.Conv4(e4) + self.rfb4_1(en[3])

            e5 = self.Maxpool4(e4)
            e5 = self.Conv5(e5) + self.rfb5_1(en[4])


        out = [e1, e2, e3, e4, e5]
        return out


class UNetDecoder(nn.Module):
    def __init__(self, n_classes=2):
        super(UNetDecoder, self).__init__()
        self.x5_dem_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.rfb2_1 = RFB_modified(128, 64)
        self.rfb3_1 = RFB_modified(256, 64)
        self.rfb4_1 = RFB_modified(512, 64)
        self.rfb5_1 = RFB_modified(1024, 64)
        self.x5_x4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

        self.x5_x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))

        self.x5_x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.x5_dem_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True))

        self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1))

        self.x5_dem_5 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(p=0.5, inplace=False)
        self.conv = nn.Conv2d(64, n_classes, kernel_size=1)
    def forward(self, features, f1='none', f2='none'):

        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        if f1 == 'none' and f2 == 'none':
            # 通道统一成64
            x5_dem_1 = self.rfb5_1(x5)
            x4_dem_1 = self.rfb4_1(x4)
            x3_dem_1 = self.rfb3_1(x3)
            x2_dem_1 = self.rfb2_1(x2)

            x5_dem_1 = self.x5_dem_1(x5_dem_1)
            x4_dem_1 = self.x4_dem_1(x4_dem_1)
            x3_dem_1 = self.x3_dem_1(x3_dem_1)
            x2_dem_1 = self.x2_dem_1(x2_dem_1)

            x5_4 = self.x5_x4(abs(F.upsample(x5_dem_1, size=x4.size()[2:], mode='bilinear') - x4_dem_1))
            x4_3 = self.x4_x3(abs(F.upsample(x4_dem_1, size=x3.size()[2:], mode='bilinear') - x3_dem_1))
            x3_2 = self.x3_x2(abs(F.upsample(x3_dem_1, size=x2.size()[2:], mode='bilinear') - x2_dem_1))
            x2_1 = self.x2_x1(abs(F.upsample(x2_dem_1, size=x1.size()[2:], mode='bilinear') - x1))

            x5_4_3 = self.x5_x4_x3(abs(F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear') - x4_3))
            x4_3_2 = self.x4_x3_x2(abs(F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear') - x3_2))
            x3_2_1 = self.x3_x2_x1(abs(F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear') - x2_1))

            x5_4_3_2 = self.x5_x4_x3_x2(abs(F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear') - x4_3_2))
            x4_3_2_1 = self.x4_x3_x2_x1(abs(F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear') - x3_2_1))

            x5_dem_4 = self.x5_dem_4(x5_4_3_2)
            x5_4_3_2_1 = self.x5_x4_x3_x2_x1(abs(F.upsample(x5_dem_4, size=x4_3_2_1.size()[2:], mode='bilinear') - x4_3_2_1))

            level4 = x5_4
            level3 = self.level3(x4_3 + x5_4_3)
            level2 = self.level2(x3_2 + x4_3_2 + x5_4_3_2)
            level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)

            x5_dem_5 = self.x5_dem_5(x5)
            output4 = self.output4(F.upsample(x5_dem_5, size=level4.size()[2:], mode='bilinear') + level4)
            output3 = self.output3(F.upsample(output4, size=level3.size()[2:], mode='bilinear') + level3)
            output2 = self.output2(F.upsample(output3, size=level2.size()[2:], mode='bilinear') + level2)
            output1 = self.output1(F.upsample(output2, size=level1.size()[2:], mode='bilinear') + level1)

            output = self.conv(output1)
            # if self.training:
            #     return output
        else:
            x5_dem_1 = self.rfb5_1(x5)
            x4_dem_1 = self.rfb4_1(x4)
            x3_dem_1 = self.rfb3_1(x3)
            x2_dem_1 = self.rfb2_1(x2)

            x5_dem_1 = self.x5_dem_1(x5_dem_1)
            x4_dem_1 = self.x4_dem_1(x4_dem_1)
            x3_dem_1 = self.x3_dem_1(x3_dem_1)
            x2_dem_1 = self.x2_dem_1(x2_dem_1)

            m1, m2, m3, m4, m5 = f1[0], f1[1], f1[2], f1[3], f1[4]
            w5, w4, w3, w2, w1 = torch.sigmoid(m5), torch.sigmoid(m4), torch.sigmoid(m3), torch.sigmoid(
                m2), torch.sigmoid(m1)  # sharpening
            w5, w4, w3, w2, w1 = w5.detach(), w4.detach(), w3.detach(), w2.detach(), w1.detach()

            x5_dem_1 = x5_dem_1 + x5_dem_1 * w5
            x5_4 = self.x5_x4(abs(F.upsample(x5_dem_1, size=x4.size()[2:], mode='bilinear') + x4_dem_1 * w4 - x4_dem_1))
            x4_3 = self.x4_x3(abs(F.upsample(x4_dem_1, size=x3.size()[2:], mode='bilinear') + x3_dem_1 * w3 - x3_dem_1))
            x3_2 = self.x3_x2(abs(F.upsample(x3_dem_1, size=x2.size()[2:], mode='bilinear') + x2_dem_1 * w2 - x2_dem_1))
            x2_1 = self.x2_x1(abs(F.upsample(x2_dem_1, size=x1.size()[2:], mode='bilinear') + x1 * w1 - x1))

            x5_4_3 = self.x5_x4_x3(abs(F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear') - x4_3))
            x4_3_2 = self.x4_x3_x2(abs(F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear') - x3_2))
            x3_2_1 = self.x3_x2_x1(abs(F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear') - x2_1))

            x5_4_3_2 = self.x5_x4_x3_x2(abs(F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear') - x4_3_2))
            x4_3_2_1 = self.x4_x3_x2_x1(abs(F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear') - x3_2_1))

            x5_dem_4 = self.x5_dem_4(x5_4_3_2)
            x5_4_3_2_1 = self.x5_x4_x3_x2_x1(
                abs(F.upsample(x5_dem_4, size=x4_3_2_1.size()[2:], mode='bilinear') - x4_3_2_1))

            x5_dem_5 = self.x5_dem_5(x5)
            level4 = x5_4
            level3 = self.level3(x4_3 + x5_4_3)
            level2 = self.level2(x3_2 + x4_3_2 + x5_4_3_2)
            level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)


            output4 = self.output4(F.upsample(x5_dem_5, size=level4.size()[2:], mode='bilinear') + level4)
            output3 = self.output3(F.upsample(output4, size=level3.size()[2:], mode='bilinear') + level3)
            output2 = self.output2(F.upsample(output3, size=level2.size()[2:], mode='bilinear') + level2)
            output1 = self.output1(F.upsample(output2, size=level1.size()[2:], mode='bilinear') + level1)

            output = self.conv(output1)
        # out_img = self.dropout(level1)
        # out_img = self.conv(out_img)

        return output, [output1, output2, output3, output4, x5_dem_5]



class DDUnet(nn.Module):
    def __init__(self, in_ch=3, n_classes=9):
        super(DDUnet, self).__init__()
        self.n_classes = n_classes
        self.encoder = UNetEncoder(in_ch)
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
    model = DDUnet().cuda()
    x = torch.randn(1, 1, 224, 224).cuda()
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

