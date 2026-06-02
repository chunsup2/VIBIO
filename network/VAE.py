import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import pooling


class VIBHO(nn.Module):
    def __init__(self, input_size=288*320, z_dim=1):
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


class VIBCNN_backup(nn.Module):
    def __init__(
        self,
        depth: int,
        z_dim: int,
        num_classes: int,
        input_height: int = 272,
        input_width: int = 320,
        pooling = "average",
        ker: int = 3,
        pad: int = 1,
        mode: str = "train",
        debug: bool = False,
    ):
        """
        depth: Number of ConvBlocks to use (each includes AvgPool2d downsampling, H,W halved per layer)
        z_dim: Latent space dimension
        num_classes: Number of classification categories
        input_height, input_width: Actual input dimensions for the network (e.g., 272x320)
        """
        super(VIBCNN_backup, self).__init__()
        self.mode = mode
        self.ker = ker
        self.pad = pad
        self.debug = debug
        self.input_height = input_height
        self.input_width = input_width
        self.pooling = pooling

        # encoder channel sizes
        channels = [16, 24, 32, 48, 64, 96, 128]
        # channels = [16, 32, 64, 128, 256, 512]
        assert depth <= len(channels) - 1, \
            f"depth={depth} is too large; maximum depth is {len(channels) - 1}"
        self.depth = depth
        self.channels = channels

        # initial conv: 1 -> 16, maintains H,W
        self.inc = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Downsampling conv blocks
        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            self.conv_layers.append(
                ConvBlock(channels[i], channels[i+1], pooling=self.pooling, ker=self.ker, pad=self.pad)
            )

        # 🔑 Auto-infer the flattened dimension of the encoder output (classifier_input_dim)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_height, input_width)
            x = self.inc(dummy)
            if self.debug:
                print(f"[DEBUG] after inc: {x.shape}")
            for li, layer in enumerate(self.conv_layers):
                x = layer(x)
                if self.debug:
                    print(f"[DEBUG] after conv_layers[{li}]: {x.shape}")

            # Store final encoder C,H,W (for debugging purposes)
            self.enc_C, self.enc_H, self.enc_W = x.shape[1:]
            classifier_input_dim = x.view(1, -1).size(1)
            if self.debug:
                print(f"[DEBUG] encoder output shape: {x.shape}")
                print(f"[DEBUG] classifier_input_dim = {classifier_input_dim}")

        # VIB bottleneck: flat -> mu/logvar -> z
        self.mu     = nn.Linear(classifier_input_dim, z_dim)
        self.logvar = nn.Linear(classifier_input_dim, z_dim)

        # classifier on z
        self.classifier = nn.Linear(z_dim, num_classes)

        # decoder...
        # Note: decoder_input output dimension must equal classifier_input_dim
        # to facilitate reshaping back to the encoder's feature map dimensions.
        self.decoder_input = nn.Linear(z_dim + num_classes, classifier_input_dim)

        # Reverse the channel order for the decoder
        channels_rev = channels[: depth+1][::-1]  # e.g., if depth=3: [64,32,16]
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
        """
        x: (B, 1, H, W), where H,W must match input_height, input_width at initialization
        label: During training, provide one-hot or multi-hot labels (dimension = num_classes)
        mode: 'train' uses reparameterization sampling; other modes use mu directly
        """
        self.mode = mode

        # encode
        if self.debug:
            print(f"[DEBUG] input: {x.shape}")

        x_enc = self.inc(x)
        if self.debug:
            print(f"[DEBUG] after inc: {x_enc.shape}")

        for li, layer in enumerate(self.conv_layers):
            x_enc = layer(x_enc)
            if self.debug:
                print(f"[DEBUG] after conv_layers[{li}]: {x_enc.shape}")

        # Store encoder output shape for the decoder
        batch_size, C, H, W = x_enc.shape
        flat = x_enc.view(batch_size, -1)  # [batch_size, 87040]

        # bottleneck
        mu     = self.mu(flat)
        logvar = self.logvar(flat)
        if mode == 'train':
            z = self.reparameterize(mu, logvar)
        else:
            z = mu  # Use mean during inference

        # classification head
        t = self.classifier(z)


        # # decode / reconstruct
        # recon = None
        if mode == 'train':
            # Use ground truth labels (one-hot) for concatenation during training
            d_in = torch.cat((z, label), dim=1)
        else:
            # Use predicted logits/labels t for concatenation during inference
            d_in = torch.cat((z, t), dim=1)

        recon = self.decoder_input(d_in)
        recon = recon.view(batch_size, C, H, W)

        for li, layer in enumerate(self.decoder_layers):
            recon = layer(recon)
            if self.debug:
                print(f"[DEBUG] after decoder_layers[{li}]: {recon.shape}")

        recon = self.decoder_output(recon)
        if self.debug:
            print(f"[DEBUG] recon output: {recon.shape}")

        return t, mu, logvar, recon


class VIBCNN(nn.Module):
    def __init__(
        self,
        depth: int,
        z_dim: int,
        num_classes: int,
        input_height: int = 272,
        input_width: int = 320,
        pooling: str = "average",
        ker: int = 3,
        pad: int = 1,
        mode: str = "train",
        debug: bool = False,
    ):
        """
        depth: Number of ConvBlocks to use (each includes AvgPool2d downsampling, H,W halved per layer)
        z_dim: Latent space dimension
        num_classes: Number of classification categories
        input_height, input_width: Actual input dimensions for the network (e.g., 272x320)
        """
        super(VIBCNN, self).__init__()
        self.mode = mode
        self.pooling = pooling
        self.ker = ker
        self.pad = pad
        self.debug = debug
        self.input_height = input_height
        self.input_width = input_width

        # encoder channel sizes
        channels = [16, 24, 32, 48, 64, 96, 128, 256, 512]
        # channels = [16, 32, 64, 128, 256, 512]
        assert depth <= len(channels) - 1, \
            f"depth={depth} is too large; maximum depth is {len(channels) - 1}"
        self.depth = depth
        self.channels = channels

        # initial conv: 1 -> 16, maintains H,W
        self.inc = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=self.ker, padding=self.pad),
            # nn.ReLU(inplace=True),
        )

        # Downsampling conv blocks
        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            self.conv_layers.append(
                ConvBlock(channels[i], channels[i+1], pooling=self.pooling, ker=self.ker, pad=self.pad)
            )

        # 🔑 Auto-infer the flattened dimension of the encoder output (classifier_input_dim)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_height, input_width)
            x = self.inc(dummy)
            if self.debug:
                print(f"[DEBUG] after inc: {x.shape}")
            for li, layer in enumerate(self.conv_layers):
                x = layer(x)
                if self.debug:
                    print(f"[DEBUG] after conv_layers[{li}]: {x.shape}")

            # Store final encoder C,H,W (for debugging purposes)
            self.enc_C, self.enc_H, self.enc_W = x.shape[1:]
            classifier_input_dim = x.view(1, -1).size(1)
            if self.debug:
                print(f"[DEBUG] encoder output shape: {x.shape}")
                print(f"[DEBUG] classifier_input_dim = {classifier_input_dim}")

        # VIB bottleneck: flat -> mu/logvar -> z
        self.mu     = nn.Linear(classifier_input_dim, z_dim)
        self.logvar = nn.Linear(classifier_input_dim, z_dim)

        # classifier on z
        self.classifier = nn.Linear(z_dim, num_classes)

        # # decoder...
        # # Note: decoder_input output dimension must equal classifier_input_dim
        # # to facilitate reshaping back to the encoder's feature map dimensions.
        # self.decoder_input = nn.Linear(z_dim + num_classes, classifier_input_dim)
        #
        # # Reverse the channel order for the decoder
        # channels_rev = channels[: depth+1][::-1]  # e.g., if depth=3: [64,32,16]
        # self.decoder_layers = nn.ModuleList()
        # for i in range(depth):
        #     in_ch  = channels_rev[i]
        #     out_ch = channels_rev[i+1]
        #     self.decoder_layers.append(nn.Sequential(
        #         nn.Upsample(scale_factor=2, mode='nearest'),
        #         nn.Conv2d(in_ch, out_ch, kernel_size=self.ker, padding=self.pad),
        #         nn.InstanceNorm2d(out_ch),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_ch, out_ch, kernel_size=self.ker, padding=self.pad),
        #         nn.InstanceNorm2d(out_ch),
        #         nn.ReLU(inplace=True),
        #     ))
        #
        # self.decoder_output = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, label=None, mode='train'):
        """
        x: (B, 1, H, W), where H,W must match input_height, input_width at initialization
        label: During training, provide one-hot or multi-hot labels (dimension = num_classes)
        mode: 'train' uses reparameterization sampling; other modes use mu directly
        """
        self.mode = mode

        # encode
        if self.debug:
            print(f"[DEBUG] input: {x.shape}")

        x_enc = self.inc(x)
        if self.debug:
            print(f"[DEBUG] after inc: {x_enc.shape}")

        for li, layer in enumerate(self.conv_layers):
            x_enc = layer(x_enc)
            if self.debug:
                print(f"[DEBUG] after conv_layers[{li}]: {x_enc.shape}")

        # Store encoder output shape for the decoder
        batch_size, C, H, W = x_enc.shape
        flat = x_enc.view(batch_size, -1)  # [batch_size, 87040]

        # bottleneck
        mu     = self.mu(flat)
        logvar = self.logvar(flat)
        if mode == 'train':
            z = self.reparameterize(mu, logvar)
        else:
            z = mu  # Use mean during inference

        # classification head
        t = self.classifier(z)


        # # decode / reconstruct
        recon = None
        # if mode == 'train':
        #     # Use ground truth labels (one-hot) for concatenation during training
        #     d_in = torch.cat((z, label), dim=1)
        # else:
        #     # Use predicted logits/labels t for concatenation during inference
        #     d_in = torch.cat((z, t), dim=1)
        #
        # recon = self.decoder_input(d_in)
        # recon = recon.view(batch_size, C, H, W)
        #
        # for li, layer in enumerate(self.decoder_layers):
        #     recon = layer(recon)
        #     if self.debug:
        #         print(f"[DEBUG] after decoder_layers[{li}]: {recon.shape}")
        #
        # recon = self.decoder_output(recon)
        # if self.debug:
        #     print(f"[DEBUG] recon output: {recon.shape}")

        return t, mu, logvar, recon





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

        
# class VIBCNN(nn.Module):
#     def __init__(self, depth, num_classes, ker=3, pad=1, mode='train', z_dim=10):
#         super(VIBCNN, self).__init__()
#         self.mode = mode
#         depth = depth - 1
#         self.ker = ker
#         self.pad = pad

#         # encoder channel sizes
#         channels = [16, 24, 32, 48, 64, 96]
#         self.conv_layers = nn.ModuleList()
#         for i in range(depth):
#             self.conv_layers.append(ConvBlock(channels[i], channels[i+1],
#                                              ker=self.ker, pad=self.pad))

#         # compute flattened feature size after conv + pooling
#         if depth == 0:
#             classifier_input_dim = 288 * 320
#         else:
#             h = 288 // (2 ** depth)
#             w = 320 // (2 ** depth)
#             classifier_input_dim = h * w * channels[depth]

#         # initial conv
#         self.inc = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )

#         # VIB bottleneck
#         self.mu     = nn.Linear(classifier_input_dim, z_dim)
#         self.logvar = nn.Linear(classifier_input_dim, z_dim)

#         # classifier on z
#         self.classifier = nn.Linear(z_dim, num_classes)
#         # self.classifier = nn.Sequential(
#         #     nn.Linear(z_dim, 64),
#         #     nn.ReLU(inplace=True),
#         #     nn.Linear(64, 32),
#         #     nn.ReLU(inplace=True),
#         #     nn.Linear(32, num_classes),
#         # )

#         # decoder...
#         self.decoder_input = nn.Linear(z_dim+num_classes, classifier_input_dim)
#         channels_rev = channels[: depth+1][::-1]
#         self.decoder_layers = nn.ModuleList()
#         for i in range(depth):
#             in_ch  = channels_rev[i]
#             out_ch = channels_rev[i+1]
#             self.decoder_layers.append(nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='nearest'),
#                 nn.Conv2d(in_ch, out_ch, kernel_size=self.ker, padding=self.pad),
#                 nn.InstanceNorm2d(out_ch),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_ch, out_ch, kernel_size=self.ker, padding=self.pad),
#                 nn.InstanceNorm2d(out_ch),
#                 nn.ReLU(inplace=True),
#             ))
#         self.decoder_output = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)
#         # self.apply(self._init_weights)
# class VIBCNN(nn.Module):
#     def __init__(self, depth, z_dim, num_classes, ker=3, pad=1, mode='train'):
#         super(VIBCNN, self).__init__()
#         self.mode = mode
#         self.ker = ker
#         self.pad = pad

#         # encoder channel sizes
#         channels = [16, 24, 32, 48, 64, 96,128]
#         self.conv_layers = nn.ModuleList()
#         for i in range(depth):
#             self.conv_layers.append(ConvBlock(channels[i], channels[i+1],
#                                              ker=self.ker, pad=self.pad))

#         # compute flattened feature size after conv + pooling
#         if depth == 0:
#             classifier_input_dim = 16 * 260 * 311
#         else:
#             h = 260 // (2 ** depth)
#             w = 311 // (2 ** depth)
#             classifier_input_dim = h * w * channels[depth]

#         # initial conv
#         self.inc = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )

#         # VIB bottleneck
#         self.mu     = nn.Linear(classifier_input_dim, z_dim)
#         self.logvar = nn.Linear(classifier_input_dim, z_dim)

#         # classifier on z
#         self.classifier = nn.Linear(z_dim, num_classes)


#         # decoder...
#         self.decoder_input = nn.Linear(z_dim+num_classes, classifier_input_dim)
#         channels_rev = channels[: depth+1][::-1]
#         self.decoder_layers = nn.ModuleList()
#         for i in range(depth):
#             in_ch  = channels_rev[i]
#             out_ch = channels_rev[i+1]
#             self.decoder_layers.append(nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='nearest'),
#                 nn.Conv2d(in_ch, out_ch, kernel_size=self.ker, padding=self.pad),
#                 nn.InstanceNorm2d(out_ch),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_ch, out_ch, kernel_size=self.ker, padding=self.pad),
#                 nn.InstanceNorm2d(out_ch),
#                 nn.ReLU(inplace=True),
#             ))
#         self.decoder_output = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)
#         # self.apply(self._init_weights)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x, label=None, mode='train'):
#         # encode
#         self.mode = mode
#         x_enc = self.inc(x)
#         for layer in self.conv_layers:
#             x_enc = layer(x_enc)

#         # remember shape for decoder
#         batch_size, C, H, W = x_enc.shape
#         flat = x_enc.view(batch_size, -1)

#         # bottleneck
#         mu     = self.mu(flat)
#         logvar = self.logvar(flat)
#         if mode == 'train':
#             z = self.reparameterize(mu, logvar)
#         else:
#             z = mu  # use mean at inference

#         # classification head
#         t = self.classifier(z)

#         # decode / reconstruct
#         if mode == 'train':
#             # print(f"z: {z}")
#             # print(f"label: {label}")
#             d_in = torch.cat((z, label), dim=1)  # concatenate z and t
#             recon = self.decoder_input(d_in)
#         else:
#             d_in = torch.cat((z, t), dim=1)  # concatenate z and t
#             recon = self.decoder_input(d_in)
#         recon = recon.view(batch_size, C, H, W)
#         for layer in self.decoder_layers:
#             recon = layer(recon)
#         recon = self.decoder_output(recon)

#         return t, mu, logvar, recon

if __name__ == '__main__':
    a = torch.randn(1, 1, 288, 320)  # 272
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
    