import torch
import torch.nn as nn
import torch.nn.functional as F

"""Dynamic Kernel ResNet"""


class AGH(nn.Module):
    def __init__(self, channels, branches=9, reduce=16, len=32):
        super(AGH, self).__init__()
        len = max(channels // reduce, len)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, len, kernel_size=(1, 1), stride=1, padding=0, bias=False),  # branches,1
            nn.BatchNorm2d(len),
            nn.ReLU(inplace=True),
        )
        self.fcs = nn.ModuleList([
            nn.Conv2d(len, channels, kernel_size=(branches, 1), stride=1, padding=0, bias=True) for _ in range(branches)
        ])
        self.softmax = nn.Softmax(dim=2)

    def forward(self, xs):  # [b,c,branches,h,w]
        attention = torch.cat([self.gap(x) for x in xs], dim=2)  # b*c*branches*1
        attention = self.fc(attention)  # ==> b*len*branches*1
        attention = torch.stack([fc(attention) for fc in self.fcs], dim=2)  # ==> b*c*branches*1*1
        attention = self.softmax(attention)

        x = torch.stack(xs, dim=2)  # ==> b*c*branches*h*w
        x = torch.sum(x * attention, dim=2)
        return x


class AGL(nn.Module):
    def __init__(self, channels, branches=9, reduce=16, len=32):
        super(AGL, self).__init__()
        len = max(channels // reduce, len)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, len, kernel_size=(branches, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(len),
            nn.ReLU(inplace=True),
        )
        self.fcs = nn.ModuleList([
            nn.Conv2d(len, channels, kernel_size=(1, 1), stride=1, padding=0, bias=True) for _ in range(branches)
        ])
        self.softmax = nn.Softmax(dim=2)

    def forward(self, xs):  # [b,c,branches,h,w]
        attention = torch.cat([self.gap(x) for x in xs], dim=2)  # b*c*branches*1
        attention = self.fc(attention)  # ==> b*len*branches*1
        attention = torch.stack([fc(attention) for fc in self.fcs], dim=2)  # ==> b*c*branches*1*1
        attention = self.softmax(attention)

        x = torch.stack(xs, dim=2)  # ==> b*c*branches*h*w
        x = torch.sum(x * attention, dim=2)
        return x


class KAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(KAConv, self).__init__()
        self.convs = nn.ModuleList([])
        for i in range(kernel_size // 2, -kernel_size // 2, -1):
            for j in range(kernel_size // 2, -kernel_size // 2, -1):
                self.convs.append(nn.Sequential(
                    nn.ConstantPad2d(padding=[j, -j, i, -i], value=0.),  # pads: left right top bottom
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ))
        self.agg = AGL(out_channels, branches=kernel_size ** 2)

    def forward(self, x):
        x = [conv(x) for conv in self.convs]
        x = self.agg(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            KAConv(mid_channels, mid_channels, kernel_size=3, stride=stride),
            # nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels == out_channels:  # when dim not change, input_features could be added diectly to out
            self.shortcut = nn.Sequential()
        else:  # when dim change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        residual = self.shortcut(residual)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += residual

        return self.relu(x)


class resnet(nn.Module):
    def __init__(self, num_classes, num_block_lists=[3, 4, 6, 3]):
        super(resnet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage_1 = self._make_layer(64, 64, 256, nums_block=num_block_lists[0], stride=1)
        self.stage_2 = self._make_layer(256, 128, 512, nums_block=num_block_lists[1], stride=2)
        self.stage_3 = self._make_layer(512, 256, 1024, nums_block=num_block_lists[2], stride=2)
        self.stage_4 = self._make_layer(1024, 512, 2048, nums_block=num_block_lists[3], stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, in_channels, mid_channels, out_channels, nums_block, stride=1):
        layers = [Bottleneck(in_channels, mid_channels, out_channels, stride=stride)]
        for _ in range(1, nums_block):
            layers.append(Bottleneck(out_channels, mid_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.basic_conv(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x


def ResNet(num_classes=1000, depth=18):
    assert depth in [50, 101, 152], 'depth invalid'
    key2blocks = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
    }
    model = resnet(num_classes, key2blocks[depth])
    return model