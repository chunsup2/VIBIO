import torch
import torch.nn as nn
import torch.nn.functional as F

class VIBHO(nn.Module):
    def __init__(self, input_size=64*64, z_dim=1):
        super(VIBHO, self).__init__()

        self.fc_mu     = nn.Linear(input_size, z_dim)
        self.fc_logvar = nn.Linear(input_size, z_dim)
        self.fc_out    = nn.Linear(z_dim, 1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)     
        mu = self.fc_mu(x)         
        logvar = self.fc_logvar(x)     
        
        z = self.reparameterize(mu, logvar)  
        t = self.fc_out(z)             
        
        return t, mu, logvar

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
    

import torch
import torch.nn as nn

class VIBCNNEstimation(nn.Module):
    def __init__(self, depth, num_classes, ker=3, pad=1, mode='train', z_dim=10,
                 est_dim=2, est_use_sigmoid=False):
        """
        CNN + VIB + 分类 + 重建，并新增“估计头 est_head”
        Args:
            depth: 与你原始实现一致（注意：内部会用 depth-1）
            num_classes: 分类类别数
            z_dim: VIB潜变量维度
            est_dim: 估计向量维度（例如 2 表示 [x, y]）
            est_use_sigmoid: 若为 True，则对估计输出做 Sigmoid（便于做 [0,1] 归一化坐标）
        """
        super(VIBCNNEstimation, self).__init__()
        self.mode = mode
        depth = depth - 1
        self.ker = ker
        self.pad = pad
        self.est_use_sigmoid = est_use_sigmoid

        # encoder channel sizes
        channels = [16, 24, 32, 48, 64, 96]
        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            self.conv_layers.append(
                ConvBlock(channels[i], channels[i+1], ker=self.ker, pad=self.pad)
            )

        # 计算展平维度（修正了 depth==0 的情况）
        if depth == 0:
            # inc -> (B,16,64,64)，因此是 64*64*16
            classifier_input_dim = 64 * 64 * 16
        else:
            h = 64 // (2 ** depth)
            w = 64 // (2 ** depth)
            classifier_input_dim = h * w * channels[depth]

        # initial conv
        self.inc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # VIB bottleneck
        self.mu     = nn.Linear(classifier_input_dim, z_dim)
        self.logvar = nn.Linear(classifier_input_dim, z_dim)

        # heads on z
        self.classifier = nn.Linear(z_dim, num_classes)  # det_head
        self.estimator  = nn.Linear(z_dim, est_dim)      # est_head
        if est_use_sigmoid:
            self.est_act = nn.Sigmoid()
        else:
            self.est_act = nn.Identity()

        # decoder（条件仍使用 z 与 label/t 拼接，保持原逻辑）
        self.decoder_input = nn.Linear(z_dim + num_classes, classifier_input_dim)
        channels_rev = channels[: depth+1][::-1]
        self.decoder_layers = nn.ModuleList()
        for i in range(depth):
            in_ch  = channels_rev[i]
            out_ch = channels_rev[i+1]
            self.decoder_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, kernel_size=self.ker, padding=self.pad),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=self.ker, padding=self.pad),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        self.decoder_output = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, label=None, mode='train'):
        self.mode = mode

        # encode
        x_enc = self.inc(x)
        for layer in self.conv_layers:
            x_enc = layer(x_enc)

        # remember shape for decoder
        batch_size, C, H, W = x_enc.shape
        flat = x_enc.view(batch_size, -1)

        # bottleneck
        mu     = self.mu(flat)
        logvar = self.logvar(flat)
        if mode == 'train':
            z = self.reparameterize(mu, logvar)
        else:
            z = mu  # inference用均值

        # heads on z
        t = self.classifier(z)                 # 分类 logits
        theta_hat = self.est_act(self.estimator(z))  # 连续估计（可选Sigmoid）

        # decode / reconstruct（保持你原来的条件拼接逻辑）
        if mode == 'train':
            if label is None:
                raise ValueError("In train mode, `label` (one-hot) must be provided for conditional decoder.")
            d_in = torch.cat((z, label), dim=1)
        else:
            d_in = torch.cat((z, t), dim=1)  # 推理时拼接 logits
        recon = self.decoder_input(d_in)
        recon = recon.view(batch_size, C, H, W)
        for layer in self.decoder_layers:
            recon = layer(recon)
        recon = self.decoder_output(recon)

        # 新增了 theta_hat 的返回
        return t, theta_hat, mu, logvar, recon


class VIBCNN(nn.Module):
    def __init__(self, depth, z_dim, num_classes, ker=3, pad=1, mode='train'):
        super(VIBCNN, self).__init__()
        self.mode = mode
        # depth = depth - 1
        self.ker = ker
        self.pad = pad
        self.z_dim = z_dim

        # encoder channel sizes
        channels = [16, 24, 32, 48, 64, 96, 128]
        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            self.conv_layers.append(ConvBlock(channels[i], channels[i+1],
                                             ker=self.ker, pad=self.pad))

        # compute flattened feature size after conv + pooling
        if depth == 0:
            classifier_input_dim =16 * 64 * 64
        else:
            h = 64 // (2 ** depth)
            w = 64 // (2 ** depth)
            classifier_input_dim = h * w * channels[depth]

        # initial conv
        self.inc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # VIB bottleneck
        self.mu     = nn.Linear(classifier_input_dim, z_dim)
        self.logvar = nn.Linear(classifier_input_dim, z_dim)

        # classifier on z
        self.classifier = nn.Linear(z_dim, num_classes)

        # decoder...
        self.decoder_input = nn.Linear(z_dim+num_classes, classifier_input_dim)
        channels_rev = channels[: depth+1][::-1]
        self.decoder_layers = nn.ModuleList()
        for i in range(depth):
            in_ch  = channels_rev[i]
            out_ch = channels_rev[i+1]
            self.decoder_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, kernel_size=self.ker, padding=self.pad),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=self.ker, padding=self.pad),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        self.decoder_output = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)
        # self.apply(self._init_weights)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, label=None, mode='train'):
        # encode
        self.mode = mode
        x_enc = self.inc(x)
        for layer in self.conv_layers:
            x_enc = layer(x_enc)

        # remember shape for decoder
        batch_size, C, H, W = x_enc.shape
        flat = x_enc.view(batch_size, -1)

        # bottleneck
        mu     = self.mu(flat)
        logvar = self.logvar(flat)
        if mode == 'train':
            z = self.reparameterize(mu, logvar)
        else:
            z = mu  # use mean at inference

        # classification head
        t = self.classifier(z)
      
        # decode / reconstruct
        if mode == 'train':
            d_in = torch.cat((z, label), dim=1)  # concatenate z and t
            recon = self.decoder_input(d_in)
        else:
            d_in = torch.cat((z, t), dim=1)  # concatenate z and t
            recon = self.decoder_input(d_in)
        recon = recon.view(batch_size, C, H, W)
        for layer in self.decoder_layers:
            recon = layer(recon)
        recon = self.decoder_output(recon)

        return t, mu, logvar, recon
    # VIB 的目标是让 z 保留对任务有用的最小充分信息

if __name__ == '__main__':
    a = torch.randn(1, 1, 64, 64)
    b = torch.randn(1, 1)
  

    model = VIBCNN(depth=6, num_classes=2)
    print(sum(p.numel() for p in model.parameters()))
    # print(model(a).shape)

    t, mu, logvar, x = model(a, torch.zeros(1, 2), mode='train')
    print(t.shape, mu.shape, logvar.shape, x.shape)