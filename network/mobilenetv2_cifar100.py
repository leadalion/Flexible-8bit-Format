import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layer_utils import *

class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual

class MobileNetV2(nn.Module):

    def __init__(self, class_num=100):
        super().__init__()

        pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        stage1 = LinearBottleNeck(32, 16, 1, 1)
        stage2 = self._make_stage(2, 16, 24, 2, 6)
        stage3 = self._make_stage(3, 24, 32, 2, 6)
        stage4 = self._make_stage(4, 32, 64, 2, 6)
        stage5 = self._make_stage(3, 64, 96, 1, 6)
        stage6 = self._make_stage(3, 96, 160, 1, 6)
        stage7 = LinearBottleNeck(160, 320, 1, 6)

        

        conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        conv2 = nn.Conv2d(1280, class_num, 1)
        # self.features = nn.Sequential(*[pre,stage1,stage2,stage3,stage4,stage5,
        # stage6,stage7,nn.Sequential(conv1, Adap_avg_pool().cuda(), conv2, Flatten().cuda())])
        self.features = nn.Sequential(*[pre,stage1,*stage2,*stage3,*stage4,*stage5,
        *stage6,stage7,conv1,nn.Sequential(Adap_avg_pool().cuda(), conv2, Flatten().cuda())])
        
        self.out_num = 1

        self.scion_block = None
        self.scion_block_pos = -1
        self.scicon_len = 0
        # self.features = nn.Sequential(*[pre,stage1,stage2,stage3,stage4,stage5,stage6,stage7])

    def forward(self, x):
        if self.scion_block is None:
            x = self.features(x)
            # x = self.conv1(x)
            # x = F.adaptive_avg_pool2d(x, 1)
            # x = self.conv2(x)
            # x = x.view(x.size(0), -1)
        elif self.scion_block_pos >= 0 and \
                self.scion_block_pos + self.scicon_len <= len(self.features):
            x = self.features[:self.scion_block_pos](x)
            x = self.scion_block(x)
            x = self.features[self.scion_block_pos
                              +self.scicon_len:](x)
        else:
            raise RuntimeError("Out of range: {}/{}!".
                               format(self.scion_block_pos,
                                      len(self.features)))

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)

    def set_scion(self, block, position, scicon_len):
        if scicon_len <= 0:
            raise RuntimeError("Illegal scicon length: {}".
                               format(scicon_len))
        if position + scicon_len - 1 > len(self.features):
            raise RuntimeError("Out of range: {}/{}!".
                               format(position,
                                      len(self.features)))
        self.scion_block = block
        self.scion_block_pos = position
        self.scicon_len = scicon_len

    def reset_scion(self):
        self.scion_block = None
        self.scion_block_pos = -1
        self.scicon_len = 0

def mobilenetv2():
    return MobileNetV2()