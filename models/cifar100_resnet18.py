from torch import nn
from flgo.utils.fmodule import FModule
import torchvision.models as models
import torch

class CIFAR100FResNet18(FModule):
    def __init__(self, num_classes=100):
        super().__init__()

        # 加载预定义的 resnet18 模型，不加载预训练权重
        self.resnet = models.resnet18(pretrained=False)

        # 修改第一层卷积，使其适应 32x32 的输入
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 将 maxpool 层替换为 Identity 层，避免过早下采样
        self.resnet.maxpool = nn.Identity()
        # 修改全连接层，输出类别数为 num_classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)


    def extract_features(self, x):
        """
        特征提取阶段：完成除全连接层外的所有前向传播操作，
        最后得到的输出经过全局平均池化并展平为特征向量。
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)  # 或 x.view(x.size(0), -1)
        return x

    def forward(self, x):
        """
        前向传播：特征提取、路径嵌入（可选）和分类。

        :param x: 输入张量，形状 [batch_size, 3, 32, 32]
        :return: 输出 logits，形状 [batch_size, num_classes]
        """
        features = self.extract_features(x)
        out = self.resnet.fc(features)
        return out


# 以下代码用于适配 flgo 框架，不需要修改
def init_local_module(object):
    pass


def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        print(object.__class__.__name__)
        object.model = CIFAR100FResNet18().to(object.device)


class CIFAR100ResNet18:
    init_local_module = init_local_module
    init_global_module = init_global_module
