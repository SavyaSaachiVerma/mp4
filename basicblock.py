import torch
import torch.nn as nn


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

        # downsample if mismatch of dimensions between residual and output
        # using 1*1 linear projection
        if self.downsample:
            self.correctedDim = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=self.stride,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            residual = self.correctedDim(residual)
        x += residual

        return x
