from torch import nn
from flgo.utils.fmodule import FModule
import torchvision.models as models
import torch


class CIFAR100FResNet18(FModule):
    def __init__(self, num_classes=100):
        super().__init__()
        # 定义一个简单的CNN结构，替代复杂的ResNet18
        # 特征提取器部分
        self.feature_extractor = nn.Sequential(
            # Block 1: 32x32 -> 14x14
            nn.Conv2d(3, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 14x14 -> 6x6
            nn.Conv2d(64, 128, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 分类头部分
        self.head = nn.Sequential(
            # 输入维度需要根据feature_extractor的输出计算：128 * 6 * 6 = 4608
            nn.Linear(128 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),  # 加入Dropout防止过拟合
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        # 展平特征
        features_flattened = torch.flatten(features, 1)
        out = self.head(features_flattened)
        return out


# 以下代码用于适配 flgo 框架，不需要修改
def init_local_module(object):
    pass


def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        # 确保服务器加载的是这个修改后的新模型
        object.model = CIFAR100FResNet18().to(object.device)


# 为了兼容性，保留这个类
class CIFAR100ResNet18:
    init_local_module = init_local_module
    init_global_module = init_global_module