import torch
import torch.nn as nn
import torch.nn.functional as F


# 输入32*32*32
class PRSNet(nn.Module):
    def __init__(self):
        super(PRSNet, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 4, 3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = nn.Conv2d(4, 8, 3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        self.conv5 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool3d(kernel_size=2)

        # 全连接层
        self.fc1_1 = nn.Linear(64, 32)
        self.fc1_2 = nn.Linear(32, 16)
        self.fc1_3 = nn.Linear(16, 4)

        self.fc2_1 = nn.Linear(64, 32)
        self.fc2_2 = nn.Linear(32, 16)
        self.fc2_3 = nn.Linear(16, 4)

        self.fc3_1 = nn.Linear(64, 32)
        self.fc3_2 = nn.Linear(32, 16)
        self.fc3_3 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.leaky_relu(self.maxpool1(self.conv1(x)))
        x = F.leaky_relu(self.maxpool2(self.conv2(x)))
        x = F.leaky_relu(self.maxpool3(self.conv3(x)))
        x = F.leaky_relu(self.maxpool4(self.conv4(x)))
        conv_result = F.leaky_relu(self.maxpool5(self.conv5(x)))

        y1 = F.leaky_relu(self.fc1_1(conv_result))
        y1 = F.leaky_relu(self.fc1_2(y1))
        y1 = F.leaky_relu(self.fc1_3(y1))

        y2 = F.leaky_relu(self.fc2_1(conv_result))
        y2 = F.leaky_relu(self.fc2_2(y2))
        y2 = F.leaky_relu(self.fc2_3(y2))

        y3 = F.leaky_relu(self.fc3_1(conv_result))
        y3 = F.leaky_relu(self.fc3_2(y3))
        y3 = F.leaky_relu(self.fc3_3(y3))

        return y1, y2, y3
