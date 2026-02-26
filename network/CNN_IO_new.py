import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ker, pad):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ker, padding=pad),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.block(x)


# class BinaryClassifier(nn.Module):
#     def __init__(self, depth, num_classes, ker=3, pad=1):
#         super(BinaryClassifier, self).__init__()
#         # channels = [4, 8, 16, 24, 32, 48, 64, 96, 128]  # Example channel sizes, modify if needed
#         # channels = [16, 32, 64, 128, 256, 512]  # Example channel sizes, modify if needed
#         channels = [16, 24, 32, 48, 64, 96]
#         self.conv_layers = nn.ModuleList()
#         for i in range(depth):
#             self.conv_layers.append(ConvBlock(channels[i], channels[i+1], ker=ker,pad=pad))
#         if depth == 0:
#             # classifier_input_dim = 4 * 260 * 311
#             classifier_input_dim = 4 * 272 * 320
#         else:
#             # classifier_input_dim = (260 // (2**depth)) * (311 // (2**depth)) * channels[depth]
#             classifier_input_dim = (272 // (2 ** depth)) * (320 // (2 ** depth)) * channels[depth]
#         self.inc = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=3, padding=1),
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(classifier_input_dim, num_classes),
#         )
#     def forward(self, x):
#         # print(x.shape)
#         x = self.inc(x)
#         for layer in self.conv_layers:
#             x = layer(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         # print()
#
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
#
#         self.classifier = nn.Sequential(
#             nn.Linear(classifier_input_dim, num_classes),
#         )
#     def forward(self, x):
#         x = self.inc(x)
#         for layer in self.conv_layers:
#             x = layer(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         return self.classifier(x)

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
        depth: Number of ConvBlocks to use (each contains an AvgPool2d for downsampling)
        num_classes: Number of output classes (2 for binary classification)
        input_height, input_width: The actual input dimensions seen by the network
            - Set to (272, 320) here because the Dataset has already been padded
        """
        super(BinaryClassifier, self).__init__()
        self.debug = debug

        # Channel configuration; your original was [16, 24, 32, 48, 64, 96, 128]
        channels = [16, 24, 32, 48, 64, 96, 128, 256, 512]
        assert depth <= len(channels) - 1, \
            f"depth={depth} 太大，最多只能到 {len(channels) - 1}"

        self.depth = depth

        # initial conv: 1 -> 16, maintains H,W
        self.inc = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)

        # Downsampling layers
        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            self.conv_layers.append(
                ConvBlock(channels[i], channels[i+1], ker=ker, pad=pad)
            )

        # 🔑 Key: Automatically infer the input dimensions for the classifier
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
        x: (B, 1, H, W), where H and W should be 272 and 320 (matching the initialization)
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


if __name__ == '__main__':
    a = torch.randn(1, 1, 288, 320)
    b = torch.randn(1, 1)
    
    model = BinaryClassifier(depth=6, num_classes=2)
    print(sum(p.numel() for p in model.parameters()))
    print(model(a).shape)