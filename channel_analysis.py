import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader

# Import your modules
# Ensure these files are accessible in the python path
from network.VAE import VIBCNN, VIBCNN_backup
from network.ResNet_IO import ResNetX
from network.CNN_IO import BinaryClassifier
from dataloader import MRIDataset1, MRIDataset2, MRIDataset3_2, MRIDataset2_2

# --- Configuration ---
# Update these paths and parameters to match your trained model
MODEL_PATH = "/home/chunsup2/Downloads/SKEBKS_MRI/1.0/VIBIO/emaVIBCNN_measure_0.001_0.005_depth3_z8_io1.0.pth"
DATA_PATH = '/shared/anastasio-s2/SI/HCP_selected/sks_3.0_0.2_25.0_c2_num_signals_diffusion_n/test/'
CLS_TYPE = 'VIBCNN'  # VIBCNN, ResNet, or CNN
DEPTH = 3  # Must match training
Z_DIM = 8  # Must match training
TRAIN_TYPE = 'measure'  # measure, cls
USE_RAM = 0  # 0 or 1, match training


def load_test_data(args):
    """Replicates the dataset loading logic from your training script"""
    if args.use_ram:
        if args.train_type == 'measure':
            test_dataset = MRIDataset2_2(args.test_image_path, proportion=1.0)
        else:
            test_dataset = MRIDataset2(args.test_image_path, proportion=1.0)
    else:
        if args.train_type == 'measure':
            # Assuming MRIDataset3_2 was used for 'measure' without RAM in training
            test_dataset = MRIDataset3_2(args.test_image_path, proportion=1.0)
        else:
            test_dataset = MRIDataset1(args.test_image_path, proportion=1.0)

    return DataLoader(test_dataset, batch_size=1, shuffle=True)


def visualize_feature_maps():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Setup Arguments (Mimicking argparse for easy configuration)
    class Args:
        pass

    args = Args()
    args.cls_type = CLS_TYPE
    args.depth = DEPTH
    args.z_dim = Z_DIM
    args.train_type = TRAIN_TYPE
    args.use_ram = USE_RAM
    args.test_image_path = DATA_PATH

    # 2. Initialize Model
    print(f"Initializing {args.cls_type}...")
    if args.cls_type == 'ResNet':
        model = ResNetX(args.depth, num_classes=2).to(device)
    elif args.cls_type == 'CNN':
        model = BinaryClassifier(args.depth, num_classes=2).to(device)
    elif args.cls_type == 'VIBCNN':
        model = VIBCNN_backup(args.depth, args.z_dim, num_classes=2).to(device)
    else:
        raise ValueError(f"Unknown cls_type {args.cls_type}")

    # 3. Load Weights
    # Handle DataParallel saving (keys might start with 'module.')
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        # Check if loading a full state dict or just parts
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Fix 'module.' prefix if it exists
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        print("Model weights loaded successfully.")
    else:
        print(f"Error: Model path {MODEL_PATH} not found.")
        return

    model.eval()

    # 4. Register Hook to Capture Features
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # --- AUTOMATIC LAYER DETECTION ---
    # This block attempts to find the last Conv2d layer automatically.
    # If this fails, uncomment the manual registration block below.
    # last_conv_layer = None
    # last_conv_name = None
    #
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         last_conv_layer = module
    #         last_conv_name = name
    #
    # if last_conv_layer:
    #     print(f"Hooking into layer: {last_conv_name}")
    #     last_conv_layer.register_forward_hook(get_activation('last_conv'))
    # else:
    #     print("Could not find a Conv2d layer automatically.")
    #     # Manual fallback (You might need to adjust 'encoder.conv4' based on print(model))
    #     # print(model)
    #     # model.encoder[-1].register_forward_hook(get_activation('last_conv'))
    target_layer = model.conv_layers[-1]
    print(f"Hooking into: model.conv_layers[-1] ({target_layer})")
    target_layer.register_forward_hook(get_activation('encoder_out'))

    # 5. Get Data and Run Inference
    test_loader = load_test_data(args)
    data_iter = iter(test_loader)
    data_batch = next(data_iter)

    # Handle data unpacking based on your script's logic
    if args.train_type == 'measure':
        measure = data_batch[0].to(device)
        task_label = data_batch[1].to(device)
        inputs = measure
    elif args.train_type == 'cls':
        image = data_batch[0].to(device)
        task_label = data_batch[2].to(device)
        inputs = image

    print(f"Input shape: {inputs.shape}")

    # Forward pass
    if args.cls_type == 'VIBCNN':
        # VIBCNN forward might return tuple (t, mu, logvar, recon)
        _ = model(inputs, mode='test')
    else:
        _ = model(inputs)

    # 6. Visualization
    if 'encoder_out' not in activation:
        print("No activation captured. Check your hook registration.")
        return

    feature_maps = activation['encoder_out'].cpu().numpy()
    # Take the first image in the batch: [C, H, W]
    feature_maps = feature_maps[0]

    num_maps = feature_maps.shape[0]
    print(f"captured feature map shape: {feature_maps.shape}")

    # Plot first 16 (or fewer) maps
    num_to_plot = min(48, num_maps)
    cols = 4
    rows = (num_to_plot + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    fig.suptitle(f'Last Encoder Layer Output (First {num_to_plot} channels)', fontsize=16)

    axes_flat = axes.flatten()
    print(num_to_plot)

    for i in range(num_to_plot):
        ax = axes_flat[i]
        # Determine if we need to use a colormap or if it's RGB
        # Usually feature maps are [H, W], so we use a colormap
        print(feature_maps[i].min(), feature_maps[i].max())
        im = ax.imshow(feature_maps[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Map {i}')

    # Hide unused subplots
    for i in range(num_to_plot, rows * cols):
        axes.flat[i].axis('off')

    plt.tight_layout()
    plt.show()
    print("Visualization complete.")


def visualize_observer_channels():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Setup Arguments
    class Args:
        pass

    args = Args()
    args.cls_type = CLS_TYPE
    args.depth = DEPTH
    args.z_dim = Z_DIM
    args.train_type = TRAIN_TYPE
    args.use_ram = USE_RAM
    args.test_image_path = DATA_PATH

    # 2. Initialize Model
    print(f"Initializing {args.cls_type}...")
    model = VIBCNN_backup(depth=args.depth, z_dim=args.z_dim, num_classes=2).to(device)

    # 3. Load Weights
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(new_state_dict)
            print("Model weights loaded.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            return
    else:
        print(f"Error: Model path {MODEL_PATH} not found.")
        return

    model.eval()

    # 4. Register Hook
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # Hook the last convolutional layer of the encoder
    target_layer = model.conv_layers[-1]
    target_layer.register_forward_hook(get_activation('encoder_out'))

    # 5. Inference
    test_loader = load_test_data(args)
    try:
        data_batch = next(iter(test_loader))
    except StopIteration:
        print("Error: Dataset empty.");
        return

    if args.train_type == 'measure':
        inputs = data_batch[0].to(device)
    elif args.train_type == 'cls':
        inputs = data_batch[0].to(device)

    _ = model(inputs, mode='test')

    if 'encoder_out' not in activation:
        print("No activation captured.");
        return

    # [C, H, W]
    feature_maps = activation['encoder_out'].cpu().numpy()[0]
    C, H, W = feature_maps.shape
    print(f"Feature Maps Shape: {feature_maps.shape}")

    # 6. CALCULATE IMPORTANCE (The "Channelized Observer" Step)
    importance_scores = calculate_channel_importance(model, (C, H, W))

    if importance_scores is None: return

    # Sort channels by importance (Descending)
    sorted_indices = np.argsort(importance_scores)[::-1]

    # 7. VISUALIZATION
    # We will create a figure with 2 parts:
    # Top: Bar chart of channel importance
    # Bottom: Grid of the Top 16 most important channels

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3])

    # --- Top: Importance Distribution ---
    ax_bar = fig.add_subplot(gs[0])
    ax_bar.bar(range(len(importance_scores)), importance_scores, color='gray', alpha=0.5)
    # Highlight the top 16
    ax_bar.bar(sorted_indices[:16], importance_scores[sorted_indices[:16]], color='red', alpha=0.7)
    ax_bar.set_title("Channel Importance Spectrum (Red = Top 16 displayed below)")
    ax_bar.set_xlabel("Channel Index")
    ax_bar.set_ylabel("Normalized Influence on Latent Space")
    ax_bar.set_xlim(0, len(importance_scores))

    # --- Bottom: Top 16 Feature Maps ---
    # Create sub-grid for the feature maps
    num_to_plot = min(16, C)
    cols = 4
    rows = (num_to_plot + cols - 1) // cols

    gs_maps = gs[1].subgridspec(rows, cols)

    for i in range(num_to_plot):
        ch_idx = sorted_indices[i]  # Get the real channel index
        score = importance_scores[ch_idx]

        ax = fig.add_subplot(gs_maps[i // cols, i % cols])

        # Plot the map
        im = ax.imshow(feature_maps[ch_idx], cmap='viridis')
        ax.axis('off')

        # Title with Importance Score
        ax.set_title(f"Ch {ch_idx}\nImp: {score:.2f}", fontsize=10)

    plt.tight_layout()
    plt.show()
    print("Analysis complete.")

    return feature_maps

if __name__ == "__main__":
    visualize_feature_maps()