import torch.nn as nn
import torch

#带归一化和激活函数的卷积操作
class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels,**kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
#多卷积核叠加操作
class Conv3d_simple(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, flag_bn, **kwargs):
        super(Conv3d_simple, self).__init__()
        self.flag = flag_bn
        self.pad = int((ksize - 1)/2)
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, ksize), padding=(0, 0, self.pad), bias=False)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1, ksize, 1), padding=(0, self.pad, 0), bias=False)
        self.conv_3 = nn.Conv3d(out_channels, out_channels, kernel_size=(ksize, 1, 1), padding=(self.pad, 0, 0), bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.PReLU(out_channels)
    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        if self.flag:
            out = self.bn(out)
        out = self.act(out)
        return out

#类似LENET-5结构，参考C3D网络
class C3D_Simple(nn.Module):
    def __init__(self, num_classes):
        super(C3D_Simple, self).__init__()
        self.conv_1 = BasicConv3d(in_channels=1, out_channels=64, kernel_size=(1,3,3))
        self.pool_1 = nn.AvgPool3d((1,2,2), (1,2,2))
        self.conv_2 = BasicConv3d(in_channels=64, out_channels=128, kernel_size=(1,3,3))
        self.pool_2 = nn.AvgPool3d((1,2,2), (1,2,2))
        self.conv_3a = BasicConv3d(in_channels=128, out_channels=256, kernel_size=3)

        self.pool_3 = nn.AvgPool3d((1,2,2), (1,2,2))
        self.conv_4a = BasicConv3d(in_channels=256, out_channels=512, kernel_size=3)

        self.pool_4 = nn.AvgPool3d(2, 2)
        self.conv_5a = BasicConv3d(in_channels=512, out_channels=256, kernel_size=3)

        self.pool_5 = nn.AdaptiveAvgPool3d(output_size=1)

        self.fc6 = nn.Linear(256, num_classes)
    def forward(self, input):
        x = self.conv_1(input)

        x = self.pool_1(x)

        x = self.conv_2(x)

        x = self.pool_2(x)

        x = self.conv_3a(x)

        x = self.pool_3(x)

        x = self.conv_4a(x)

        x = self.pool_4(x)

        x = self.conv_5a(x)

        x = self.pool_5(x)

        x = x.view(x.shape[0], -1)

        x = self.fc6(x)

        return x

#参考C3D网络，加入downsample的shortcut通道，使用更小的卷积叠加操作
class C3D_ResNet(nn.Module):
    def __init__(self, num_classes):
        super(C3D_ResNet, self).__init__()
        self.conv_1_1 = BasicConv3d(in_channels=1, out_channels=128, kernel_size=(3, 1, 1))
        self.conv_1_2 = BasicConv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 1))
        self.conv_1_3 = BasicConv3d(in_channels=128, out_channels=128, kernel_size=(1, 1, 3))
        self.pool_1 = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

        self.conv_2_1 = BasicConv3d(in_channels=128, out_channels=256, kernel_size=(3, 1, 1))
        self.conv_2_2 = BasicConv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 1))
        self.conv_2_3 = BasicConv3d(in_channels=256, out_channels=256, kernel_size=(1, 1, 3))
        self.pool_2 = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

        self.conv_3_1 = BasicConv3d(in_channels=256, out_channels=256, kernel_size=(3, 1, 1))
        self.conv_3_2 = BasicConv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 1))
        self.conv_3_3 = BasicConv3d(in_channels=256, out_channels=256, kernel_size=(1, 1, 3))

        self.conv_4_1 = BasicConv3d(in_channels=256, out_channels=128, kernel_size=(3, 1, 1))
        self.conv_4_2 = BasicConv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 1))
        self.conv_4_3 = BasicConv3d(in_channels=128, out_channels=128, kernel_size=(1, 1, 3))
        self.pool_4 = nn.AvgPool3d(2, 2)

        self.downsample = nn.AdaptiveAvgPool3d(output_size=(24, 26, 26))

        self.conv_5_1 = BasicConv3d(in_channels=128, out_channels=64, kernel_size=(3, 1, 1))
        self.conv_5_2 = BasicConv3d(in_channels=64, out_channels=64, kernel_size=(1, 3, 1))
        self.conv_5_3 = BasicConv3d(in_channels=64, out_channels=64, kernel_size=(1, 1, 3))

        self.conv_6_1 = BasicConv3d(in_channels=64, out_channels=32, kernel_size=(3, 1, 1))
        self.conv_6_2 = BasicConv3d(in_channels=32, out_channels=32, kernel_size=(1, 3, 1))
        self.conv_6_3 = BasicConv3d(in_channels=32, out_channels=32, kernel_size=(1, 1, 3))

        self.pool_6 = nn.AdaptiveAvgPool3d(output_size=1)

        self.fc6 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout3d(p=0.5, inplace=True)
    def forward(self, input):
        # 32 x 128 x 128
        x = self.conv_1_1(input)
        # 30 x 128 x 128
        x = self.conv_1_2(x)
        # 30 x 126 x 128
        x = self.conv_1_3(x)
        # 30 x 126 x 126
        x_conv1 = self.pool_1(x)
        # 30 x 63 x 63
        x = self.conv_2_1(x_conv1)
        # 30 x 63 x 63
        x = self.conv_2_2(x)
        # 30 x 61 x 63
        x = self.conv_2_3(x)
        # 30 x 61 x 61
        x = self.pool_2(x)
        # 30 x 30 x 30
        x = self.conv_3_1(x)
        # 28 x 30 x 30
        x = self.conv_3_2(x)
        # 28 x 28 x 30
        x = self.conv_3_3(x)
        # 28 x 28 x 28
        x = self.conv_4_1(x)
        # 26 x 28 x 28
        x = self.conv_4_2(x)
        # 26 x 26 x 28
        x = self.conv_4_3(x)
        # 26 x 26 x 26
        x_downsample = self.downsample(x_conv1)
        # 26 x 26 x 26
        x = x + x_downsample
        if self.training:
            x = self.dropout(x)
        # 26 x 26 x 26
        x = self.pool_4(x)
        # 13 x 13 x 13
        x = self.conv_5_1(x)
        # 11 x 13 x 13
        x = self.conv_5_2(x)
        # 11 x 11 x 13
        x = self.conv_5_3(x)
        # 11 x 11 x 11
        x = self.conv_6_1(x)
        # 9 x 11 x 11
        x = self.conv_6_2(x)
        # 9 x 9 x 11
        x = self.conv_6_3(x)
        # 9 x 9 x 9
        x = self.pool_6(x)
        if self.training:
            x = self.dropout(x)
        # 1 x 1 x 1
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)

        return x

#参考U-net网络，增加多个shortcut通道，使用小卷积的叠加操作
class C3D_ResNet_simple(nn.Module):
    def __init__(self, num_classes):
        super(C3D_ResNet_simple, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv3d(1, 16, (1, 1, 1), padding=0),
            nn.Conv3d(16, 16, (1, 3, 1), padding=(0, 1, 0)),
            nn.Conv3d(16, 16, (1, 1, 3), padding=(0, 0, 1)),
            nn.PReLU(16),

            nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(16, 16, (1, 3, 1), padding=(0, 1, 0)),
            nn.Conv3d(16, 16, (1, 1, 3), padding=(0, 0, 1)),
            nn.PReLU(16),
        )

        self.layer_2 = nn.Sequential(
            nn.Conv3d(32, 32, (3, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(32, 32, (1, 3, 1), padding=(0, 1, 0)),
            nn.Conv3d(32, 32, (1, 1, 3), padding=(0, 0, 1)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (3, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(32, 32, (1, 3, 1), padding=(0, 1, 0)),
            nn.Conv3d(32, 32, (1, 1, 3), padding=(0, 0, 1)),
            nn.PReLU(32),

            nn.Conv3d(32, 32, (3, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(32, 32, (1, 3, 1), padding=(0, 1, 0)),
            nn.Conv3d(32, 32, (1, 1, 3), padding=(0, 0, 1)),
            nn.PReLU(32),
        )

        self.layer_3 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(64, 64, (1, 3, 1), padding=(0, 1, 0)),
            nn.Conv3d(64, 64, (1, 1, 3), padding=(0, 0, 1)),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (3, 1, 1), padding=(2, 0, 0), dilation=(2, 1, 1)),
            nn.Conv3d(64, 64, (1, 3, 1), padding=(0, 2, 0), dilation=(1, 2, 1)),
            nn.Conv3d(64, 64, (1, 1, 3), padding=(0, 0, 2), dilation=(1, 1, 2)),
            nn.PReLU(64),

            nn.Conv3d(64, 64, (3, 1, 1), padding=(4, 0, 0), dilation=(4, 1, 1)),
            nn.Conv3d(64, 64, (1, 3, 1), padding=(0, 4, 0), dilation=(1, 4, 1)),
            nn.Conv3d(64, 64, (1, 1, 3), padding=(0, 0, 4), dilation=(1, 1, 4)),
            nn.PReLU(64),
        )

        self.layer_4 = nn.Sequential(
            nn.Conv3d(128, 128, (3, 1, 1), padding=(3, 0, 0), dilation=(3, 1, 1)),
            nn.Conv3d(128, 128, (1, 3, 1), padding=(0, 3, 0), dilation=(1, 3, 1)),
            nn.Conv3d(128, 128, (1, 1, 3), padding=(0, 0, 3), dilation=(1, 1, 3)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (3, 1, 1), padding=(4, 0, 0), dilation=(4, 1, 1)),
            nn.Conv3d(128, 128, (1, 3, 1), padding=(0, 4, 0), dilation=(1, 4, 1)),
            nn.Conv3d(128, 128, (1, 1, 3), padding=(0, 0, 4), dilation=(1, 1, 4)),
            nn.PReLU(128),

            nn.Conv3d(128, 128, (3, 1, 1), padding=(5, 0, 0), dilation=(5, 1, 1)),
            nn.Conv3d(128, 128, (1, 3, 1), padding=(0, 5, 0), dilation=(1, 5, 1)),
            nn.Conv3d(128, 128, (1, 1, 3), padding=(0, 0, 5), dilation=(1, 1, 5)),
            nn.PReLU(128),
        )
        self.layer_5 = nn.Sequential(
            nn.Conv3d(128, 256, (3, 1, 1), padding=(1, 0, 0))
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32),
        )
        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64),
        )
        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128),
        )
        # self.down_conv4 = nn.Sequential(
        #     nn.Conv3d(128, 256, 2, 2),
        #     nn.PReLU(256)
        # )
        self.drop = nn.Dropout(0.3, True)
        self.map = nn.Sequential(
            nn.Conv3d(128, 64, 1, 1),
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
            nn.Sigmoid(),
        )
        self.out = nn.Linear(64, num_classes)

    def forward(self, input):
        long_range1 = self.layer_1(input)
        long_range1 += input

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.layer_2(short_range1) + short_range1
        if self.training:
            long_range2 = self.drop(long_range2)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.layer_3(short_range2) + short_range2
        if self.training:
            long_range3 = self.drop(long_range3)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.layer_4(short_range3) + short_range3
        if self.training:
            long_range4 = self.drop(long_range4)

        out = self.map(long_range4)
        out = out.view(out.shape[0], -1)
        out = self.out(out)

        return out

#参考inception结构，通过cat来融合多种不同尺寸卷积操作的特征映射
class C3D_ResNet_inception(nn.Module):
    def __init__(self, num_classes):
        super(C3D_ResNet_inception, self).__init__()

        self.layer_1_1 = nn.Conv3d(1, 16, 1, padding=0, bias=False)
        self.layer_1_3 = Conv3d_simple(1, 16, 3, True)
        self.layer_1_7 = Conv3d_simple(1, 16, 7, True)

        self.layer_2_1 = nn.Conv3d(97, 128, 1, padding=0, bias=False)
        self.layer_2_3 = Conv3d_simple(97, 128, 3, True)
        self.layer_2_7 = Conv3d_simple(97, 128, 7, True)

        self.layer_3_1 = nn.Conv3d(481, 128, 1, padding=0, bias=False)
        self.layer_3_3 = Conv3d_simple(481, 128, 3, False)
        self.layer_3_7 = Conv3d_simple(481, 128, 7, False)

        self.layer_4_1 = nn.Conv3d(865, 128, 1, padding=0, bias=False)
        self.layer_4_3 = Conv3d_simple(865, 128, 3, False)

        self.layer_5_1 = nn.Conv3d(1121, 64, 1, padding=0, bias=False)
        self.layer_5_3 = Conv3d_simple(1121, 64, 3, False)

        self.layer_6_3 = Conv3d_simple(1249, 128, 3, False)

        self.layer_7_3 = Conv3d_simple(128, 128, 3, False)

        self.layer_8_3 = Conv3d_simple(128, 128, 3, False)

        self.drop = nn.Dropout3d(p=0.25, inplace=False)

        self.out = nn.Linear(128, num_classes)
        self.pool_simple = nn.AvgPool3d((1, 2, 2))
        self.pool_full = nn.AvgPool3d((2, 2, 2))

    def forward(self, input):
        l1_1 = self.layer_1_1(input)
        l1_3 = self.layer_1_3(input)
        l1_7 = self.layer_1_7(input)
        l1_out = torch.cat((l1_1, l1_3, l1_7, input), 1)
        # 97 x 32 x 128 x 128
        l1_out = self.pool_simple(l1_out)
        # 97 x 32 x 64 x 64
        l2_1 = self.layer_2_1(l1_out)
        l2_3 = self.layer_2_3(l1_out)
        l2_7 = self.layer_2_7(l1_out)
        l2_out = torch.cat((l2_1, l2_3, l2_7, l1_out), 1)
        # 481 x 32 x 64 x 64
        l2_out = self.pool_simple(l2_out)
        # 481 x 32 x 32 x 32
        l3_1 = self.layer_3_1(l2_out)
        l3_3 = self.layer_3_3(l2_out)
        l3_7 = self.layer_3_7(l2_out)
        l3_out = torch.cat((l3_1, l3_3, l3_7, l2_out), 1)
        # 865 x 32 x 32 x 32
        l3_out = self.pool_full(l3_out)
        # 865 x 16 x 16 x 16
        l4_1 = self.layer_4_1(l3_out)
        l4_3 = self.layer_4_3(l3_out)
        l4_out = torch.cat((l4_1, l4_3, l3_out), 1)
        # 1121 x 16 x 16 x 16
        l4_out = self.pool_full(l4_out)
        # 1121 x 8 x 8 x 8
        l5_1 = self.layer_5_1(l4_out)
        l5_3 = self.layer_5_3(l4_out)
        l5_out = torch.cat((l5_1, l5_3, l4_out), 1)
        # 1249 x 8 x 8 x 8
        l6 = self.layer_6_3(l5_out)
        # 128 x 8 x 8 x 8
        l6 = self.pool_full(l6)
        # 128 x 4 x 4 x 4
        l7 = self.layer_7_3(l6)
        l7 = self.pool_full(l7)
        # 128 x 2 x 2 x 2
        l8 = self.layer_8_3(l7)
        l8 = self.pool_full(l8)
        # 128 x 1 x 1 x 1
        l8 = l8.view(input.shape[0], -1)

        out = self.out(l8)
        return out

#基本3x3尺寸的卷积操作
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

#基本shortcut结构，默认是加上原输入
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

#基本shortcut结构，默认是加上原输入，多一层卷积操作，产生更多的通道
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

#基于ResNet(2D)相似结构，替换网络中所有2D操作为3D操作。
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.pool_ada = nn.AdaptiveAvgPool3d(output_size=1)

        self.fc = nn.Linear(512, num_classes)
        # self.fc_for_dist = nn.Linear(1, num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool_ada(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



