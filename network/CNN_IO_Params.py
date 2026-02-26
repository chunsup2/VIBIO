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

class BinaryClassifier(nn.Module):
    def __init__(self, depth, num_classes, ker=3, pad=1):
        super(BinaryClassifier, self).__init__()
        depth = depth - 1
        channels = [16, 24, 32, 48, 64, 96]  # 可按需调整
        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            self.conv_layers.append(ConvBlock(channels[i], channels[i+1], ker=ker, pad=pad))

        if depth == 0:
            classifier_input_dim = 288 * 320
        else:
            classifier_input_dim = (288 // (2**depth)) * (320 // (2**depth)) * channels[depth]

        self.inc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            # 可选激活/归一化
        )

        # 简单线性分类头
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, num_classes),
        )

    def forward(self, x):
        x = self.inc(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)


# ---------------- 参数量与MACs/FLOPs统计 ----------------
def count_params(model):
    """可训练参数量（个）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def count_macs(model, input_size=(1, 1, 288, 320), include_act=False):
    """
    统计单次前向的 MACs（乘加次数，~ FLOPs/2）。
    - 支持 Conv2d(含depthwise/groups)、Linear、AvgPool2d、(Instance)Norm、ReLU、AdaptiveAvgPool2d。
    - include_act=True 时把 ReLU 也按一次“操作/元素”计入（通常不计）。
    """
    macs = 0
    handles = []

    def conv_hook(module, inp, out):
        nonlocal macs
        # Conv2d: MACs = Cout*Hout*Wout * (Cin/groups)*Kh*Kw
        x = inp[0]
        B, Cin, Hin, Win = x.shape
        B, Cout, Hout, Wout = out.shape
        Kh, Kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        groups = module.groups
        macs += Cout * Hout * Wout * (Cin // groups) * Kh * Kw

    def linear_hook(module, inp, out):
        nonlocal macs
        in_features = module.in_features
        out_features = module.out_features
        macs += in_features * out_features

    def bn_in_hook(module, inp, out):
        nonlocal macs
        # 归一化/仿射近似为 1 次乘加/元素
        macs += out.numel()

    def relu_hook(module, inp, out):
        nonlocal macs
        if include_act:
            macs += out.numel()

    def avgpool_hook(module, inp, out):
        nonlocal macs
        # AvgPool2d: 每个输出元素 ~ Kh*Kw 次加法 + 1 次除法（近似按 Kh*Kw 计）
        x = inp[0]
        B, C, Hin, Win = x.shape
        B, C, Hout, Wout = out.shape
        Kh, Kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        if Kh is None or Kw is None:  # 防御
            Kh = Kw = 1
        macs += C * Hout * Wout * Kh * Kw

    def adapavgpool_hook(module, inp, out):
        nonlocal macs
        # AdaptiveAvgPool2d(1): 每通道做 H*W 次加法（近似）
        x = inp[0]
        B, C, H, W = x.shape
        macs += C * H * W

    # 注册 hooks
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            handles.append(m.register_forward_hook(bn_in_hook))
        elif isinstance(m, nn.ReLU):
            handles.append(m.register_forward_hook(relu_hook))
        elif isinstance(m, nn.AvgPool2d):
            handles.append(m.register_forward_hook(avgpool_hook))
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            handles.append(m.register_forward_hook(adapavgpool_hook))

    # 前向一次
    device = next(model.parameters()).device
    x = torch.randn(*input_size, device=device)
    _ = model(x)

    for h in handles:
        h.remove()

    return macs


if __name__ == '__main__':
    a = torch.randn(1, 1, 288, 320)
    model = BinaryClassifier(depth=6, num_classes=2)
    model.eval()

    # 参数量
    params = count_params(model)
    print(f"Trainable params: {params:,}")

    # 前向与输出维度
    with torch.no_grad():
        out = model(a)
    print("Output shape:", tuple(out.shape))

    # 计算 MACs（单张图像）
    macs = count_macs(model, input_size=(1, 1, 288, 320), include_act=False)
    print(f"MACs per image: {macs/1e6:.3f} MMACs  (~ {2*macs/1e9:.3f} GFLOPs)")

    # 你原始打印
    # print(sum(p.numel() for p in model.parameters()))
    # print(model(a).shape)
