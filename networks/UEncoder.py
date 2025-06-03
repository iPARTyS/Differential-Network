import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1, activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):

    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),  # conv3x3 + BN + ReLU
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False)  # conv3x3 + BN
        )
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)    # conv1x1 + BN + ReLU

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


class DWCONV(nn.Module):
    """
    Depthwise Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=None):
        super(DWCONV, self).__init__()
        if groups == None:
            groups = in_channels
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups, bias=True
        )

    def forward(self, x):
        result = self.depthwise(x)
        return result


class UEncoder(nn.Module):

    def __init__(self):
        super(UEncoder, self).__init__()
        self.res1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.res3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.res4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.res5 = DoubleConv(512, 1024)
        self.pool5 = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        x = self.res1(x)
        features.append(x)  # (224, 224, 64)
        x = self.pool1(x)   # (112, 112, 64)

        x = self.res2(x)
        features.append(x)  # (112, 112, 128)
        x = self.pool2(x)   # (56, 56, 128)

        x = self.res3(x)
        features.append(x)  # (56, 56, 256)
        x = self.pool3(x)   # (28, 28, 256)

        x = self.res4(x)
        features.append(x)  # (28, 28, 512)
        x = self.pool4(x)   # (14, 14, 512)

        x = self.res5(x)
        features.append(x)  # (14, 14, 1024)
        x = self.pool5(x)  # (7, 7, 1024)
        features.append(x)

        return features

