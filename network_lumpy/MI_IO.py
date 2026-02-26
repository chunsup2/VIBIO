import torch
import torch.nn as nn
import torch.nn.functional as F

class MConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ker, pad):
        super(MConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ker, padding=pad),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=ker, padding=pad),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)
    
class MIEstimator(nn.Module):
    def __init__(self, input_size, depth, ker=3, pad=1):
        super(MIEstimator, self).__init__()
        self.depth = depth
        self.input_size = input_size
        channels = [8, 16, 32, 64, 128]
        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            self.conv_layers.append(MConvBlock(channels[i], channels[i+1], ker=ker, pad=pad))
        
        if depth == 0:
            classifier_input_dim = 64 * 64
        else:
            # Dynamically calculate the classifier input dimensions based on input image size and network depth
            classifier_input_dim = (64 // (2**depth)) * (64 // (2**depth)) * channels[depth]
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 1),
        )
        self.inc = nn.Sequential(
            nn.Conv2d(input_size, 4, kernel_size=3, padding=1),
        )

    def forward(self, x, y):
        y = torch.repeat_interleave(y, self.input_size, dim=1)
        y = y.unsqueeze(2).unsqueeze(3)
        y = y.expand(-1,-1,x.shape[2],x.shape[3])
        x = self.inc(x)
        y = self.inc(y)
        z = F.relu(torch.cat([x, y], dim=1))
        for layer in self.conv_layers:
            z = layer(z)
        z = z.view(z.size(0), -1)
        return self.classifier(z)

if __name__ == '__main__':
    a = torch.randn(1, 1, 64, 64)
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