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
    def __init__(self, in_channels, out_channels, pooling="average", ker=3, pad=1):
        super(ConvBlock, self).__init__()
        if pooling == 'max':
            pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pooling == 'average':
            pool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(f"Invalid pooling type: {pooling}. Use 'max' or 'average'.")

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ker, padding=pad),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            pool_layer  # Use the selected layer here
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
        pooling: str = "average",
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
            f"depth={depth} is too large; the maximum allowed depth is {len(channels) - 1}"

        self.depth = depth

        # initial conv: 1 -> 16, maintains H,W
        self.inc = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)

        # Downsampling layers
        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            self.conv_layers.append(
                ConvBlock(channels[i], channels[i+1], pooling=pooling, ker=ker, pad=pad)
            )

        # Automatically infer the input dimensions for the classifier
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
    a = torch.randn(1, 1, 288, 320)  # 272
    b = torch.randn(1, 1)
    
    model = BinaryClassifier(depth=6, num_classes=2)
    print(sum(p.numel() for p in model.parameters()))
    print(model(a).shape)