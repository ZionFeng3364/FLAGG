from torch import nn
from flgo.utils.fmodule import FModule
import torchvision.models as models
import torch


class CIFAR10FResNet18(FModule):
    def __init__(self, num_classes=10):
        super().__init__()

        # 加载预定义的 resnet18 模型，不加载预训练权重
        resnet = models.resnet18(pretrained=False)

        # 1. 定义特征提取器 (feature_extractor)
        # 它包含 ResNet 中除了最后一个全连接层之外的所有层
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 修改第一层以适应CIFAR10
            resnet.bn1,
            resnet.relu,
            nn.Identity(),  # 替换原来的 maxpool
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )

        # 2. 定义分类头 (head)
        # 它就是原来的全连接层
        self.head = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        """
        前向传播：先通过特征提取器，然后展平，最后通过分类头。
        """
        features = self.feature_extractor(x)
        # 在特征提取器和分类头之间需要一个展平操作
        features_flattened = torch.flatten(features, 1)
        out = self.head(features_flattened)
        return out


# 以下代码用于适配 flgo 框架，不需要修改
def init_local_module(object):
    pass


def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = CIFAR10FResNet18().to(object.device)


# 为了兼容性，保留这个类
class CIFAR10ResNet18:
    init_local_module = init_local_module
    init_global_module = init_global_module