import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 32

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNetX(num_block, num_classes=2):
    return ResNet(BasicBlock, [num_block, num_block], num_classes=num_classes)


if __name__ == '__main__':
    a = torch.randn(1, 1, 288, 320)
    b = torch.randn(1, 1)
    # model = UNet_Tiny_woc(1, 1)
    # parameters
    # print(sum(p.numel() for p in model.parameters()))
    # print(model(a).shape)

    # model = UNet_Small(1, 1)
    # print(sum(p.numel() for p in model.parameters()))
    # print(model(a).shape)

    # model = UNet_Tiny(1, 1)
    # print(sum(p.numel() for p in model.parameters()))
    # print(model(a).shape)

    # model = ResNetX(5)
    # print(sum(p.numel() for p in model.parameters()))
    # print(model(a).shape)

    # model = MIEstimator(depth=3, input_size=1)
    # print(sum(p.numel() for p in model.parameters()))
    # print(model(a, b).shape)

    # RDN model
    # model = RDN(scale_factor=1, num_channels=1, num_features=64, growth_rate=64, num_blocks=6, num_layers=8)
    # print(sum(p.numel() for p in model.parameters()))

    # model = BinaryClassifier(depth=5, num_classes=2)
    # print(sum(p.numel() for p in model.parameters()))

    model = VIBCNN(depth=5, num_classes=2)
    print(sum(p.numel() for p in model.parameters()))
    # print(model(a).shape)

    t, mu, logvar, x = model(a, torch.zeros(1, 2), mode='train')
    print(t.shape, mu.shape, logvar.shape, x.shape)