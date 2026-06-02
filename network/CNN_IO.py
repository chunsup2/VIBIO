import torch
import torch.nn as nn
import torch.nn.functional as F






import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ker=3, pad=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ker, padding=pad),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)  # 每层 /2
        )

    def forward(self, x):
        return self.block(x)


class BinaryClassifier(nn.Module):
    def __init__(
        self,
        depth: int,
        num_classes: int,
        input_height: int = 272,
        input_width: int = 320,
        ker: int = 3,
        pad: int = 1,
        debug: bool = False,
    ):
        """
        depth: 使用多少个 ConvBlock（每个都有 AvgPool2d 下采样）
        num_classes: 输出类别数（你是二分类就是 2）
        input_height, input_width: 网络实际看到的输入尺寸
            - 这里是 (272, 320)，因为 Dataset 已经 pad 过
        """
        super(BinaryClassifier, self).__init__()
        self.debug = debug

        # 通道配置，你原来的是 [16, 24, 32, 48, 64, 96, 128]
        channels = [16, 24, 32, 48, 64, 96, 128]
        assert depth <= len(channels) - 1, \
            f"depth={depth} 太大，最多只能到 {len(channels) - 1}"

        self.depth = depth

        # 第一层：从 1 通道到 16 通道，不改变尺寸
        self.inc = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)

        # 下采样层
        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            self.conv_layers.append(
                ConvBlock(channels[i], channels[i + 1], ker=ker, pad=pad)
            )

        # 🔑 关键：自动推理 classifier 的输入维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_height, input_width)
            x = self.inc(dummy)
            if self.debug:
                print(f"[DEBUG] after inc: {x.shape}")  # (1, 16, H, W)

            for li, layer in enumerate(self.conv_layers):
                x = layer(x)
                if self.debug:
                    print(f"[DEBUG] after conv_layers[{li}]: {x.shape}")

            classifier_input_dim = x.view(1, -1).size(1)
            if self.debug:
                print(f"[DEBUG] classifier_input_dim = {classifier_input_dim}")

        self.classifier = nn.Linear(classifier_input_dim, num_classes)

    def forward(self, x):
        """
        x: (B, 1, H, W)，H,W 应该是 272,320（或者和初始化时一致）
        """
        if self.debug:
            print(f"[DEBUG] input: {x.shape}")   # (B, 1, 272, 320)

        x = self.inc(x)
        if self.debug:
            print(f"[DEBUG] after inc: {x.shape}")

        for li, layer in enumerate(self.conv_layers):
            x = layer(x)
            if self.debug:
                print(f"[DEBUG] after conv_layers[{li}]: {x.shape}")

        x = x.view(x.size(0), -1)
        if self.debug:
            print(f"[DEBUG] after flatten: {x.shape}")  # (B, classifier_input_dim)

        out = self.classifier(x)
        if self.debug:
            print(f"[DEBUG] output: {out.shape}")       # (B, num_classes)

        return out





# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, ker, pad):
#         super(ConvBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=ker, padding=pad),
#             nn.InstanceNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=ker, padding=pad),
#             nn.InstanceNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(kernel_size=2, stride=2)
#         )
#     def forward(self, x):
#         return self.block(x)

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, ker, pad):
#         super(ConvBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=ker, padding=pad),
#             nn.InstanceNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(kernel_size=2, stride=2)
#         )
#     def forward(self, x):
#         return self.block(x)


# class BinaryClassifier(nn.Module):
#     def __init__(self, depth, num_classes,ker=3, pad=1):
#         super(BinaryClassifier, self).__init__()
#         channels = [16, 24, 32, 48, 64, 96, 128]  # Example channel sizes, modify if needed
#         self.conv_layers = nn.ModuleList()
#         for i in range(depth):
#             self.conv_layers.append(ConvBlock(channels[i], channels[i+1], ker=ker,pad=pad))
#         if depth == 0:
#             classifier_input_dim = 16 * 260 * 311
#         else:
#             classifier_input_dim = (260 // (2**depth)) * (311 // (2**depth)) * channels[depth]
#         self.inc = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(classifier_input_dim, num_classes),
#         )
#     def forward(self, x):
#         x = self.inc(x)
#         for layer in self.conv_layers:
#             x = layer(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         return self.classifier(x)


# class BinaryClassifier(nn.Module):
#     def __init__(self, depth, num_classes,ker=3, pad=1):
#         super(BinaryClassifier, self).__init__()
#         depth = depth - 1
#         channels = [16, 24, 32, 48, 64, 96]  # Example channel sizes, modify if needed
#         self.conv_layers = nn.ModuleList()
#         for i in range(depth):
#             self.conv_layers.append(ConvBlock(channels[i], channels[i+1], ker=ker,pad=pad))
#         if depth == 0:
#             classifier_input_dim = 288 * 320
#         else:
#             classifier_input_dim = (288 // (2**depth)) * (320 // (2**depth)) * channels[depth]
#         self.inc = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             # F.relu,
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(classifier_input_dim, num_classes),
#         )
#     def forward(self, x):
#         x = self.inc(x)
#         for layer in self.conv_layers:
#             x = layer(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         return self.classifier(x)


if __name__ == '__main__':
    a = torch.randn(1, 1, 288, 320)
    b = torch.randn(1, 1)
    
    model = BinaryClassifier(depth=6, num_classes=2)
    print(sum(p.numel() for p in model.parameters()))
    print(model(a).shape)