import torch
import torch.nn as nn
import torch.nn.functional as F

class SLNNHO(nn.Module):
    def __init__(self):
        super(SLNNHO, self).__init__()
        self.fc = nn.Linear(64*64, 1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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