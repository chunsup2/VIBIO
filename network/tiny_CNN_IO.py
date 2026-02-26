import torch
import torch.nn as nn

# ---- 可分离卷积模块：3x3 depthwise + 1x1 pointwise ----
class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm='bn', act=True):
        super().__init__()
        Norm = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d}[norm]
        layers = [
            nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p,
                      groups=in_ch, bias=False),            # depthwise
            Norm(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),    # pointwise
            Norm(out_ch),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# ---- 轻量主干 ----
class LightTinyNet(nn.Module):
    """
    适合小数据集的超轻量网络
    - in_ch: 输入通道（灰度=1）
    - num_classes: 类别数
    - width_mult: 调整整体通道规模(0.5/0.75/1.0等)，越小越轻
    - norm: 'bn'（batch≥8更稳更快）或 'in'（医学小batch友好）
    """
    def __init__(self, in_ch=1, num_classes=2, width_mult=0.75, norm='in', dropout=0.1):
        super().__init__()
        def C(c):  # 通道缩放并对齐到8，保证高效
            v = int(c * width_mult)
            v = max(8, (v + 7) // 8 * 8)
            return v

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, C(16), 3, 2, 1, bias=False),   # 288x320 -> 144x160
            (nn.BatchNorm2d if norm=='bn' else nn.InstanceNorm2d)(C(16)),
            nn.ReLU(inplace=True),
        )

        self.stage = nn.Sequential(
            DWConvBlock(C(16), C(24), s=1, norm=norm),     # 144x160
            DWConvBlock(C(24), C(24), s=2, norm=norm),     # 72x80
            DWConvBlock(C(24), C(32), s=1, norm=norm),     # 72x80
            DWConvBlock(C(32), C(48), s=2, norm=norm),     # 36x40
            DWConvBlock(C(48), C(64), s=2, norm=norm),     # 18x20
            DWConvBlock(C(64), C(96), s=2, norm=norm),     # 9x10
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # GAP -> (B, C, 1, 1)
            nn.Flatten(),
            nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(C(96), num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage(x)
        x = self.head(x)
        return x


# ---------------- 参数量与MACs/FLOPs统计 ----------------
def count_params(model):
    """可训练参数量（个）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def count_macs(model, input_size=(1, 1, 288, 320)):
    """
    统计单次前向的 MACs（乘加次数，~ FLOPs/2）。
    支持 Conv2d（含depthwise）、BN/IN/ReLU、AdaptiveAvgPool2d(1)、Linear。
    """
    macs = 0
    handles = []

    def conv_hook(module, inp, out):
        nonlocal macs
        x = inp[0]
        Cout, Hout, Wout = out.shape[1], out.shape[2], out.shape[3]
        Cin = module.in_channels
        Kh, Kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        groups = module.groups
        # MACs: Cout*Hout*Wout * (Cin/groups)*Kh*Kw
        conv_macs = Cout * Hout * Wout * (Cin // groups) * Kh * Kw
        macs += conv_macs

    def linear_hook(module, inp, out):
        nonlocal macs
        in_features = module.in_features
        out_features = module.out_features
        macs += in_features * out_features

    def bn_in_hook(module, inp, out):
        nonlocal macs
        # 归一化/仿射看作 ~1 次乘加/元素
        macs += out.numel()

    def relu_hook(module, inp, out):
        nonlocal macs
        # ReLU 视为一次比较/元素（不计入乘加，可选）
        # 如需并入统计可解注下一行：
        # macs += out.numel()
        pass

    def gap_hook(module, inp, out):
        nonlocal macs
        x = inp[0]
        # AdaptiveAvgPool2d(1): 每个通道做一次全局平均，近似 H*W 次加法/通道
        B, C, H, W = x.shape
        macs += C * H * W

    # 注册hooks
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            handles.append(m.register_forward_hook(bn_in_hook))
        elif isinstance(m, nn.ReLU):
            handles.append(m.register_forward_hook(relu_hook))
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            handles.append(m.register_forward_hook(gap_hook))

    # 前向一次
    device = next(model.parameters()).device
    x = torch.randn(*input_size, device=device)
    _ = model(x)

    # 移除hooks
    for h in handles:
        h.remove()

    return macs

# ---------------- 使用示例 ----------------
if __name__ == "__main__":
    model = LightTinyNet(in_ch=1, num_classes=2, width_mult=0.75, norm='in', dropout=0.1)
    model.eval()

    inp_size = (1, 1, 288, 320)  # (B, C, H, W)，统计时建议用 batch=1
    with torch.no_grad():
        out = model(torch.randn(*inp_size))
    params = count_params(model)
    macs = count_macs(model, inp_size)

    print("Output shape:", tuple(out.shape))
    print(f"Trainable params: {params:,}")
    # 约定：1 GMAC ≈ 2 GFLOPs
    print(f"MACs per image: {macs/1e6:.3f} MMACs  (~ {2*macs/1e9:.3f} GFLOPs)")
