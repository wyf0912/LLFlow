import torch
import torch.nn as nn
import torch.nn.functional as F

class MSIA(nn.Module):
    def __init__(self, filters, activation='lrelu'):
        super().__init__()
        # Down 1
        self.conv_bn_relu_1 = Conv_BN_Relu(filters, activation)
        # Down 2
        self.down_2 = MaxPooling2D(2, 2)
        self.conv_bn_relu_2 = Conv_BN_Relu(filters, activation)
        self.deconv_2 = ConvTranspose2D(filters, filters)
        # Down 4
        self.down_4 = MaxPooling2D(2, 2)
        self.conv_bn_relu_4 = Conv_BN_Relu(filters, activation, kernel=1)
        self.deconv_4_1 = ConvTranspose2D(filters, filters)
        self.deconv_4_2 = ConvTranspose2D(filters, filters)
        # output
        self.out = Conv2D(filters*4, filters)

    def forward(self, R, I_att):
        R_att = R * I_att
        # Down 1
        msia_1 = self.conv_bn_relu_1(R_att)
        # Down 2
        down_2 = self.down_2(R_att)
        conv_bn_relu_2 = self.conv_bn_relu_2(down_2)
        msia_2 = self.deconv_2(conv_bn_relu_2)
        # Down 4
        down_4 = self.down_4(down_2)
        conv_bn_relu_4 = self.conv_bn_relu_4(down_4)
        deconv_4 = self.deconv_4_1(conv_bn_relu_4)
        msia_4 = self.deconv_4_2(deconv_4)
        # concat
        concat = torch.cat([R, msia_1, msia_2, msia_4], dim=1)
        out = self.out(concat)
        return out


class Conv_BN_Relu(nn.Module):
    def __init__(self, channels, activation='lrelu', kernel=3):
        super().__init__()
        self.ActivationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.ActivationLayer = nn.ReLU(inplace=True)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel, padding=kernel//2),
            nn.BatchNorm2d(channels, momentum=0.99),  # 原论文用的tf.layer的默认参数
            self.ActivationLayer,
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.doubleconv = nn.Sequential(
            Conv2D(in_channels, out_channels, activation),
            Conv2D(out_channels,out_channels, activation)
        )

    def forward(self, x):
        return self.doubleconv(x)

class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.8)
        self.cbam = CBAM(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.8)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        x1 = self.relu(bn1)
        cbam = self.cbam(x1)
        conv2 = self.conv2(cbam)
        bn2 = self.bn1(conv2)
        out = bn2 + x
        return out

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu', stride=1):
        super().__init__()
        self.ActivationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.ActivationLayer = nn.ReLU(inplace=True)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            self.ActivationLayer,
        )

    def forward(self, x):
        return self.conv_relu(x)


class ConvTranspose2D(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.ActivationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.ActivationLayer = nn.ReLU(inplace=True)
        self.deconv_relu = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            self.ActivationLayer,
        )

    def forward(self, x):
        return self.deconv_relu(x)


class MaxPooling2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.maxpool(x)


class AvgPooling2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.avgpool(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x) * x
        out = self.sa(x) * x
        return x


class Concat(nn.Module):
    def forward(self, x, y):
        _, _, xh, xw = x.size()
        _, _, yh, yw = y.size()
        diffY = xh - yh
        diffX = xw - yw
        y = F.pad(y, (diffX // 2, diffX - diffX//2,
                      diffY // 2, diffY - diffY//2))
        return torch.cat((x, y), dim=1)