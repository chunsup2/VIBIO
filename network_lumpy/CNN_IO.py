import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ker, pad):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ker, padding=pad),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=ker, padding=pad),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.block(x)

# import torch
# import torch.nn as nn

# class ConvBlockPool(nn.Module):
#     """
#     带池化的 Block：Conv + IN + ReLU + Conv + IN + ReLU + MaxPool
#     用在前几层做“降采样 + 提取粗特征”
#     """
#     def __init__(self, in_channels, out_channels, ker, pad):
#         super(ConvBlockPool, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=ker, padding=pad),
#             nn.InstanceNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=ker, padding=pad),
#             nn.InstanceNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             # 用 MaxPool 比 AvgPool 更能保住局部“亮点/小信号”
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#     def forward(self, x):
#         return self.block(x)


# class ConvBlock(nn.Module):
#     """
#     不带池化的 Block：Conv + IN + ReLU + Conv + IN + ReLU
#     用在较深层，只增加表达能力，不再丢空间分辨率
#     """
#     def __init__(self, in_channels, out_channels, ker, pad):
#         super(ConvBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=ker, padding=pad),
#             nn.InstanceNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=ker, padding=pad),
#             nn.InstanceNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.block(x)


# class BinaryClassifier(nn.Module):
#     def __init__(self, depth, num_classes, ker=3, pad=1):
#         super(BinaryClassifier, self).__init__()

#         # 通道配置，可以根据需要改
#         channels = [16, 24, 32, 48, 64, 96, 192]

#         # 输入卷积：1×64×64 → 16×64×64
#         self.inc = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#         )

#         # 最多只在前 max_pools 层做下采样，后面只做卷积、不再减小分辨率
#         max_pools = 2  # 也可以改成 1 或 3 自己试
#         self.conv_layers = nn.ModuleList()
#         for i in range(depth):
#             in_ch = channels[i]
#             out_ch = channels[i + 1]
#             if i < max_pools:
#                 # 前几层：带池化的 block（分辨率减半）
#                 self.conv_layers.append(ConvBlockPool(in_ch, out_ch, ker=ker, pad=pad))
#             else:
#                 # 后几层：不带池化的 block（分辨率保持不变）
#                 self.conv_layers.append(ConvBlock(in_ch, out_ch, ker=ker, pad=pad))

#         # 计算全连接层的输入维度
#         if depth == 0:
#             # 只有 inc: 输出 16×64×64
#             spatial = 64
#             last_channels = 16
#         else:
#             # 实际发生的池化次数 = min(depth, max_pools)
#             n_pools = min(depth, max_pools)
#             spatial = 64 // (2 ** n_pools)           # 空间尺寸
#             last_channels = channels[depth]          # 最后一层的通道数

#         classifier_input_dim = spatial * spatial * last_channels

#         # 分类器
#         self.classifier = nn.Sequential(
#             nn.Linear(classifier_input_dim, num_classes),
#         )

#     def forward(self, x):
#         x = self.inc(x)           # (B, 1, 64, 64) → (B, 16, 64, 64)
#         for layer in self.conv_layers:
#             x = layer(x)          # 依次经过若干个 block

#         x = x.view(x.size(0), -1) # Flatten
#         out = self.classifier(x)  # 线性分类
#         return out

class BinaryClassifier_MultiTask(nn.Module):
    """
    CNN-based detection + estimation network
      detection head: classification (logits for CrossEntropyLoss)
      estimation head: continuous regression (e.g. θ = [x, y])
    """
    def __init__(self, depth, num_classes=2, est_dim=2, ker=3, pad=1):
        super().__init__()
        depth = depth - 1
        channels = [16, 24, 32, 48, 64, 96]
        self.conv_layers = nn.ModuleList()

        # backbone
        self.inc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True)
        )
        for i in range(depth):
            self.conv_layers.append(ConvBlock(channels[i], channels[i + 1], ker=ker, pad=pad))

        # 全局池化使输入维度自动适配
        # final_ch = channels[depth] if depth > 0 else 16
        # self.gap = nn.AdaptiveAvgPool2d(1)

        if depth == 0: 
            classifier_input_dim = 64 * 64 
        
        else: 
            classifier_input_dim = (64 // (2**depth)) * (64 // (2**depth)) * channels[depth]

        # 两个 head
        self.det_head = nn.Linear(classifier_input_dim, num_classes)  # 分类 (CrossEntropyLoss)
        self.est_head = nn.Linear(classifier_input_dim, est_dim)      # 回归 (估计)

    def forward(self, x):
        x = self.inc(x)
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(x.size(0), -1) # Flatten   

        # B, C, H, W -> B, C
        # x = self.gap(x).flatten(1)

        # 两个输出
        det_logits = self.det_head(x)  # shape: (B, num_classes)
        theta_hat = self.est_head(x)   # shape: (B, est_dim)
        return det_logits, theta_hat

class BinaryClassifier(nn.Module):
    def __init__(self, depth, num_classes, ker=3, pad=1):
        super(BinaryClassifier, self).__init__()
        
        # Use the original depth (don't subtract 1)
        channels = [16, 24, 32, 48, 64, 96, 128]  # Example channel sizes, modify if needed
        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            self.conv_layers.append(ConvBlock(channels[i], channels[i+1], ker=ker, pad=pad))
        
        # Correct the classifier input dimension calculation
        if depth == 0:
            classifier_input_dim = 16 * 64 * 64  # No convolution, input size remains 64x64
        else:
            classifier_input_dim = (64 // (2**depth)) * (64 // (2**depth)) * channels[depth]
        
        # Input convolution layer
        self.inc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, num_classes),
        )

    def forward(self, x):
        x = self.inc(x)
        for layer in self.conv_layers:
            x = layer(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)


# class BinaryClassifier(nn.Module):
#     def __init__(self, depth, num_classes,ker=3, pad=1):
#         super(BinaryClassifier, self).__init__()
#         depth = depth - 1
#         channels = [16, 24, 32, 48, 64, 96, 192]  # Example channel sizes, modify if needed
#         self.conv_layers = nn.ModuleList()
#         for i in range(depth):
#             self.conv_layers.append(ConvBlock(channels[i], channels[i+1], ker=ker,pad=pad))
#         if depth == 0:
#             classifier_input_dim = 64 * 64
#         else:
#             classifier_input_dim = (64 // (2**depth)) * (64 // (2**depth)) * channels[depth]
#         self.inc = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             # F.relu,
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(classifier_input_dim, num_classes),
#             # nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.inc(x)
#         for layer in self.conv_layers:
#             x = layer(x)
        
#         x = x.view(x.size(0), -1)  # Flatten
        
#         return self.classifier(x)


if __name__ == '__main__':
    a = torch.randn(1, 1, 64, 64)
    b = torch.randn(1, 1)

    model = BinaryClassifier(depth=6, num_classes=2)
    print(sum(p.numel() for p in model.parameters()))
    b = model(a)
