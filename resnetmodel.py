import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, basic_block, num_basic_blocks_list, num_classes=100, linear_layer_num_input=256, max_pool_stride=1):
        super(ResNet, self).__init__()
        self.basic_block = basic_block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self.conv2_x = self.add_basic_blocks(32, 32, num_basic_blocks_list[0])
        self.conv3_x = self.add_basic_blocks(32, 64, num_basic_blocks_list[1], stride=2)
        self.conv4_x = self.add_basic_blocks(64, 128, num_basic_blocks_list[2], stride=2)
        self.conv5_x = self.add_basic_blocks(128, 256, num_basic_blocks_list[3], stride=2)
        self.max_pool = nn.MaxPool2d(max_pool_stride, stride=1)
        self.fc = nn.Linear(linear_layer_num_input, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.max_pool(x)
        # need to flatten out inputs for nn.Linear
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    def add_basic_blocks(self, in_channels,  out_channels, num_basic_blocks, stride=1):

        downsample = False
        # only need to downsample in the first basic block of the group
        # as after that the in_channels = out_channels and stride=1
        if in_channels != out_channels or stride != 1:
            downsample = True

        basic_block_layers = []
        basic_block_layers.append(self.basic_block(in_channels, out_channels, stride,
                                                  downsample))

        for i in range(1, num_basic_blocks):
            basic_block_layers.append(self.basic_block(out_channels, out_channels))

        return nn.Sequential(*basic_block_layers)










