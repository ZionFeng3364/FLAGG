import torch
import torch.nn as nn
import torch.nn.functional as F
from flgo.utils.fmodule import FModule


class FEMNISTModelF(FModule):
    def __init__(self, num_classes=None):
        super().__init__()

        # 使用2D卷积来处理输入
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)

        # 池化层
        self.pool = nn.MaxPool2d(2)

        # 添加一个特征降维层，将 64*7*7=3136 降至 512
        self.feature_reduction = nn.Linear(64 * 7 * 7, 512)

        # 全连接层
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.5)

    def extract_features(self, x):
        """特征提取部分"""
        # 处理输入形状 [B, 1, 784, 1] -> [B, 1, 28, 28]
        if x.dim() == 4 and x.shape[2] == 784 and x.shape[3] == 1:
            x = x.squeeze(3).view(-1, 1, 28, 28)
        # 处理输入形状 [B, 784, 1] -> [B, 1, 28, 28]
        elif x.dim() == 3 and x.shape[1] == 784 and x.shape[2] == 1:
            x = x.squeeze(2).view(-1, 1, 28, 28)
        # 处理输入形状 [B, 784] -> [B, 1, 28, 28]
        elif x.dim() == 2 and x.shape[1] == 784:
            x = x.view(-1, 1, 28, 28)

        # 标准CNN处理
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 64, 7, 7]
        x = torch.flatten(x, 1)  # -> [B, 64*7*7]

        # 特征降维，将 3136 维降至 512 维
        x = F.relu(self.feature_reduction(x))  # -> [B, 512]

        return x

    def forward(self, x):
        """完整的前向传播"""
        features = self.extract_features(x)
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 以下代码用于适配 flgo 框架，不需要修改
def init_local_module(object):
    pass


def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        print(object.__class__.__name__)
        object.model = FEMNISTModelF().to(object.device)


class FEMNISTModel:
    init_local_module = init_local_module
    init_global_module = init_global_module