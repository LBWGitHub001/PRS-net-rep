import torch
import torch.nn as nn
import torch.nn.functional as F


# 输入32*32*32
# 模型的输出是三个平面的隐式表示(ax, by, cz, d)
class PRSNet(nn.Module):
    def __init__(self):
        super(PRSNet, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv3d(1, 4, 3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = nn.Conv3d(4, 8, 3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = nn.Conv3d(8, 16, 3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        self.conv5 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool3d(kernel_size=2)

        # 全连接层
        self.plane_predictor = Predictor()
        self.axe_predictor = Predictor()


    def forward(self, x):
        x = F.leaky_relu(self.maxpool1(self.conv1(x)))
        x = F.leaky_relu(self.maxpool2(self.conv2(x)))
        x = F.leaky_relu(self.maxpool3(self.conv3(x)))
        x = F.leaky_relu(self.maxpool4(self.conv4(x)))
        x = F.leaky_relu(self.maxpool5(self.conv5(x)))
        planes = self.plane_predictor(x)
        axes = self.axe_predictor(x)

        # plane: 只归一化法向量 (nx,ny,nz)，d 自由
        raw_norm = torch.norm(planes[..., :3], dim=-1, keepdim=True).clamp(min=1e-8)  # (B, 3, 1)
        n = planes[..., :3] / raw_norm
        d = planes[..., 3:4] / raw_norm
        planes = torch.cat([n, d], dim=-1)

        # quaternion: 整组归一化到单位四元数
        axes = F.normalize(axes, dim=-1)  # (B, 3, 4)  单位四元数

        return planes, axes

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
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
        conv_result = x.view(x.size(0), -1)

        y1 = F.leaky_relu(self.fc1_1(conv_result))
        y1 = F.leaky_relu(self.fc1_2(y1))
        y1 = self.fc1_3(y1)

        y2 = F.leaky_relu(self.fc2_1(conv_result))
        y2 = F.leaky_relu(self.fc2_2(y2))
        y2 = self.fc2_3(y2)

        y3 = F.leaky_relu(self.fc3_1(conv_result))
        y3 = F.leaky_relu(self.fc3_2(y3))
        y3 = self.fc3_3(y3)

        return torch.stack((y1, y2, y3), dim=1)