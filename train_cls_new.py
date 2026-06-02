import os
os.environ["WANDB__SERVICE_WAIT"] = "120"

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from network.UNet import UNet
from network.CNN_IO_new import BinaryClassifier
from network.VAE import VIBCNN, VIBHO
from network.HO import SLNNHO
from network.ResNet_IO import ResNetX
import sys

from dataloader import *
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

import numpy as np
import argparse
import wandb
import h5py

import webdataset as wds

from wandb import Image

from utils import load_model

from sklearn.metrics import roc_auc_score, RocCurveDisplay  # Modified import
import matplotlib.pyplot as plt
import random

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)   # Use fixed weight initialization to minimize large discrepancies.


def kl_beta_schedule(step, beta_max=0.02, ramp_start=1000, ramp_end=5000):
    """Linear warm-up from 0 to beta_max between ramp_start and ramp_end steps."""
    if step < ramp_start:
        return 0.0
    elif step < ramp_end:
        return beta_max * (step - ramp_start) / (ramp_end - ramp_start)
    else:
        return beta_max


# def compute_kl(mu, logvar, free_bits=0.0):
#     kl_per_dim = 0.5 * (mu**2 + logvar.exp() - logvar - 1)
#     kl = torch.sum(torch.clamp(kl_per_dim, min=free_bits), dim=1).mean()
#     return kl

def compute_kl(mu, logvar,
               free_bits: float = 0.5,
               clamp_logvar: bool = True,
               logvar_min: float = -20.0,
               logvar_max: float = 20.0):
    """
    mu, logvar: [B, D]
    free_bits: per-dim min KL (prevents KL from collapsing information), 0 means off
    """
    if clamp_logvar:
        logvar = torch.clamp(logvar, min=logvar_min, max=logvar_max)

    # per-dim KL: 0.5 * (mu^2 + sigma^2 - 1 - log sigma^2)
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)  # [B, D]

    # per-dim free bits (optional)
    if free_bits > 0.0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    # Sum over all dimensions, then average over the batch
    kl = torch.sum(kl_per_dim, dim=1).mean()  # scalar

    # Note: Here **no / latent_dim**, making it easier to see the real impact of dimensions on KL
    return kl


def apply_kspace_noise(imgs, mask, noise_level):
    """
    Applies complex noise in the k-space (frequency domain) to a batch of images.
    Expects imgs shape: (B, H, W) or (B, C, H, W)
    """
    # Perform batched 2D FFT
    kspace = torch.fft.fft2(imgs)

    # Generate batched complex noise
    noise_real = torch.randn_like(imgs) * noise_level
    noise_imag = torch.randn_like(imgs) * noise_level
    noise = torch.complex(noise_real, noise_imag)

    # Add noise and apply mask
    if mask is not None:
        kspace_noisy = (kspace + noise) * mask
    else:
        kspace_noisy = (kspace + noise)

    # Perform batched 2D Inverse FFT
    img_recon = torch.fft.ifft2(kspace_noisy)
    return torch.real(img_recon)


def train(args):
    torch.backends.cudnn.benchmark = True
    
    device = torch.device(args.device)

    def pick_random_coord(sample):
        """
        Takes a single unbatched sample (image, coords_array).
        Returns (image, single_chosen_coordinate).
        """
        image, coords = sample

        # Randomly select one index from the available coordinates
        rand_idx = random.randint(0, len(coords) - 1)

        # Extract that specific [x0, y0] pair
        x0, y0 = coords[rand_idx]

        # Return the image and a fixed-size tensor of [x0, y0]
        return image, torch.tensor([x0, y0], dtype=torch.float32)

    if args.train_type == 'recon':
        model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
        
        # Load the model and use multi-GPU
        if args.data_parallel:
            model = nn.DataParallel(model)
        print(device)

        # for re-train?
        # load_model(model, os.path.join(args.save_model_path, args.srtype) + '/srnet.pth')

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    if args.cls_type == 'ResNet':
        cls_model = ResNetX(args.depth, num_classes=2).to(device)
    elif args.cls_type == 'CNN':
        cls_model = BinaryClassifier(args.depth, num_classes=2, pooling=args.pooling).to(device)
    elif args.cls_type == 'VIBCNN':
        cls_model = VIBCNN(args.depth, args.z_dim, num_classes=2, pooling=args.pooling).to(device)
    else:
        raise ValueError(f"Unknown cls_type {args.cls_type}")

    if args.data_parallel:
        cls_model = nn.DataParallel(cls_model)

    # Optimizer and scheduler for cls_model
    optimizer = optim.Adam(list(cls_model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_gamma, patience=args.scheduler_patience, verbose=True)

    # Loss functions
    # criterion = nn.MSELoss().to(device)
    cls_criterion = nn.CrossEntropyLoss().to(device)  # CrossEntropyLoss expects integer labels

    total_images = int(args.n_data * 1000)

    if args.n_data < 10:
        n_shuffle = total_images
    else:
        n_shuffle = 10000

    train_dataset = (
        wds.WebDataset(args.train_image_path, shardshuffle=True)
        .slice(total_images)
        .shuffle(n_shuffle, initial=10000)  # Shuffle the shards and maintain a local buffer of 10000 samples
        .decode("torch")
        .to_tuple("image.npy", "coords.npy")  # Extract the "jpg" and "cls" keys we defined during writing
        .map(pick_random_coord)
        .batched(args.batch_size)  # Batch them together (e.g., batch size 64)
        .with_length(int(total_images / args.batch_size))  # 672
    )

    val_dataset = (
        wds.WebDataset(args.val_image_path, shardshuffle=False)
        .decode("torch")
        .to_tuple("npy", "cls")  # Extract the "jpg" and "cls" keys we defined during writing
        .batched(200)  # Batch them together (e.g., batch size 64)
        .with_length(50)
    )

    H, W = 260, 311
    target_size = 272, 320

    X, Y = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device='cuda'),
        torch.arange(W, dtype=torch.float32, device='cuda'),
        indexing='ij'
    )
    # distance = torch.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    # within_3sigma = distance <= 3 * args.sigma
    #
    # signal = torch.zeros((H, W), dtype=torch.float32, device='cuda')
    # signal[within_3sigma] = args.amplitude * torch.exp(
    #     -0.5 * ((X[within_3sigma] - x0) ** 2 + (Y[within_3sigma] - y0) ** 2) / (args.sigma ** 2)
    # )

    train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=None,
                                  prefetch_factor=args.num_workers if args.num_workers > 0 else 2)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=None,
                                prefetch_factor=4 if args.num_workers > 0 else None)

    ## --- Training State Variables ---
    global_step = 0
    best_auc = 0.5
    best_val_loss = float('inf')

    # Early stopping variables
    steps_since_improvement = 0

    temp_tensor = torch.empty(int(18 * 1024**3 / 4)).to('cuda')

    for epoch in tqdm(range(args.epochs)):
        cls_model.train()

        for i, data in enumerate(tqdm(train_dataloader)):
            images = data[0].to('cuda')
            coords_batch = data[1].to('cuda')

            B, H, W = images.shape
            half_B = B // 2

            # ---------------------------------------------------------
            # STEP A: Create and Add Signal to the FIRST HALF
            # (Do this before padding so your y0, x0 coordinates stay accurate to the anatomy!)
            # ---------------------------------------------------------

            # Inject the localized signal into the first half of the batch
            if args.signal_location == 'ske':
                x0 = torch.full((half_B,), 170.0, device='cuda')
                y0 = torch.full((half_B,), 220.0, device='cuda')

                x0 = x0.view(half_B, 1, 1)
                y0 = y0.view(half_B, 1, 1)
            else:
                # x0 = torch.tensor([meta["x0"] for meta in data[1][:half_B]], device='cuda', dtype=torch.float32)
                # y0 = torch.tensor([meta["y0"] for meta in data[1][:half_B]], device='cuda', dtype=torch.float32)
                x0 = coords_batch[:half_B, 0].view(half_B, 1, 1)
                y0 = coords_batch[:half_B, 1].view(half_B, 1, 1)

            distance_sq = (X - x0) ** 2 + (Y - y0) ** 2
            within_3sigma = distance_sq <= (3 * args.sigma) ** 2

            gaussian = args.amplitude * torch.exp(-0.5 * distance_sq / (args.sigma ** 2))
            signal = torch.where(within_3sigma, gaussian, torch.zeros_like(gaussian))

            # signal = torch.zeros((H, W), dtype=torch.float32, device='cuda')
            # signal[within_3sigma] = args.amplitude * torch.exp(-0.5 * distance_sq[within_3sigma] / (args.sigma ** 2))
            images[:half_B] = images[:half_B] + signal

            target_H, target_W = target_size
            pad_h = target_H - H
            pad_w = target_W - W

            pad_tuple = (
                max(0, pad_w // 2),
                max(0, pad_w - pad_w // 2),
                max(0, pad_h // 2),
                max(0, pad_h - pad_h // 2)
            )
            # The empty padded areas are currently 0.0
            images = F.pad(images, pad_tuple, mode="constant", value=0.0)
            images = images.unsqueeze(1)

            if args.blur_sigma > 0.0:
                import torchvision.transforms.functional as TF
                # IMPORTANT: Prevent blurring across the batch dimension (axis 0).
                # We apply 0 blur to the batch axis, and blur_sigma to H and W.

                radius = int(4.0 * args.blur_sigma + 0.5)
                kernel_size = [2 * radius + 1, 2 * radius + 1]
                sigma_list = [args.blur_sigma, args.blur_sigma]

                blurred_images = TF.gaussian_blur(images, kernel_size=kernel_size, sigma=sigma_list)

                zeros_images = torch.zeros_like(images)
                raw_noise = apply_kspace_noise(zeros_images, None, args.noise_level)

                target_std = torch.std(raw_noise, unbiased=False)

                blurred_noise = TF.gaussian_blur(raw_noise, kernel_size=kernel_size, sigma=sigma_list)

                current_std = torch.std(blurred_noise, unbiased=False)
                calibrated_noise = blurred_noise * (target_std / current_std)

                feats = blurred_images + calibrated_noise

            else:
            # Add noise
                feats = apply_kspace_noise(images, None, args.noise_level)

            task_label = torch.zeros(B, dtype=torch.long, device='cuda')
            task_label[:half_B] = 1

            # Remove one-hot encoding; ensure task_label is LongTensor
            # task_label = task_label.long()

            if args.cls_type in ['CNN', 'ResNet']:
                cls = cls_model(feats)
                loss = cls_criterion(cls, task_label)
            
            elif args.cls_type == 'VIBCNN':
                # one hot the task_label: 0 -> [0, 1], 1 -> [1, 0]
                label_one_hot = F.one_hot(task_label, num_classes=2).float()

                t, mu, logvar, recon = cls_model(feats, label=label_one_hot)
                # recon_loss = F.mse_loss(recon, feats)
                
                # KL divergence computation
                if args.anneal:
                    current_kl_weight = kl_beta_schedule(global_step, beta_max=args.kl,
                                                       ramp_start=args.ramp_start, ramp_end=args.ramp_end)
                    kl_loss = compute_kl(mu, logvar, free_bits=args.free_bits)
                else:
                    current_kl_weight = args.kl
                    # To do
                    # kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
                    # kl_loss = torch.sum(torch.clamp(kl_per_dim, min=args.free_bits), dim=1).mean()
                    # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                    kl_loss = compute_kl(mu, logvar, free_bits=args.free_bits)
                cls_loss = cls_criterion(t, task_label)

                # print('cls_loss: ', cls_loss)
                # print('kl_loss: ', kl_loss)

                loss = cls_loss + current_kl_weight * kl_loss
            else:
                raise ValueError("Model type not handled")
                   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if scheduler_flag == 1:
            #     scheduler.step()

        # ============================================================
        # VALIDATION & EARLY STOPPING (Iteration-based)
        # ============================================================
        # if global_step % args.val_interval == 0:
        # Log training metrics
        if args.cls_type in ['CNN', 'ResNet']:
            wandb.log({
                "Train Loss": loss.item(),
                # "Train Recon Loss": recon_loss.item(),
                # "Train KL Loss": args.kl * kl_loss.item(),
                # "Train HO Loss": ho_loss.item(),
                # "Train Class Loss": cls_loss.item(),
            })
        else:
            wandb.log({
                "Train Loss": loss.item(),
                # "Train Recon Loss": recon_loss.item(),
                "Train KL Loss": args.kl * kl_loss.item(),
                # "Train HO Loss": ho_loss.item(),
                "Train Class Loss": cls_loss.item(),
            })
        # print(f"\n[Step {global_step}] Starting validation...")

        # Test
        cls_model.eval()
        running_val_loss = 0.0
        val_batch_count = 0

        true_labels = []
        predicted_probs = []
        predicted_labels = []  # To store predicted class indices


        with torch.no_grad():
            for i, data in enumerate(tqdm(val_dataloader)):
                feats = data[0].to('cuda')
                feats = feats.unsqueeze(1)

                task_label = torch.tensor([int(lbl) for lbl in data[1]]).to('cuda')

                if args.cls_type in ['CNN', 'ResNet']:
                    cls = cls_model(feats)
                elif args.cls_type == 'VIBCNN':
                    t, mu, logvar, recon = cls_model(feats, mode='test')
                    cls = t

                batch_loss = cls_criterion(cls, task_label)
                running_val_loss += batch_loss.item()
                val_batch_count += 1

                # cls_loss = cls_criterion(cls, task_label)

                # For AUC calculation
                cls_probs = F.softmax(cls, dim=1)
                _, predicted = torch.max(cls_probs, 1)

                predicted_labels.extend(predicted.detach().cpu().numpy())
                predicted_probs.extend(cls_probs.detach().cpu().numpy())

                    # total += task_label.size(0)
                    # correct += (predicted == task_label).sum().item()

                # elif args.cls_type in ['HO', 'VIBHO']:
                #     test_stat = cls.squeeze()
                #     predicted = torch.where(test_stat > 0.5,
                #                           torch.ones_like(test_stat),
                #                           torch.zeros_like(test_stat)).long()
                #     predicted_labels.extend(predicted.detach().cpu().numpy())
                #     predicted_probs.extend(test_stat.detach().cpu().numpy())

                true_labels.extend(task_label.detach().cpu().numpy())


        # Convert true_labels and predicted_probs to appropriate format
        num_classes = 2

        avg_val_loss = running_val_loss / val_batch_count if val_batch_count > 0 else float('inf')

        true_labels_array = np.array(true_labels)
        predicted_probs_array = np.array(predicted_probs)

        # Handle different classifier types
        # if args.cls_type == 'CNN' or args.cls_type == 'ResNet' or args.cls_type == 'VIBCNN':
        if args.cls_type in ['CNN', 'ResNet', 'VIBCNN']:
            if num_classes == 2:
                # Binary classification: use probabilities for positive class
                roc_auc = roc_auc_score(true_labels_array, predicted_probs_array[:, 1])
            else:
                # Multi-class: use one-vs-rest approach
                roc_auc = roc_auc_score(true_labels_array, predicted_probs_array, multi_class='ovr', average='macro')
        else:  # HO or VIBHO
            roc_auc = roc_auc_score(true_labels_array, predicted_probs_array)

        if roc_auc < 0.5: roc_auc = 1 - roc_auc

        scheduler.step(roc_auc)

        # --- Saving Helper ---
        state_dict = cls_model.module.state_dict() if hasattr(cls_model, 'module') else cls_model.state_dict()
        base_path = args.save_model_path
        fname_core = f"{args.cls_type}_{args.train_type}_{args.param_setting}"
        # fname_core = f"{args.cls_type}_{args.train_type}_kl{args.kl}_lr{args.lr}_d{args.depth}_z{args.z_dim}"

        def save_ckpt(suffix):
            torch.save(state_dict, f"{base_path}/{fname_core}_{suffix}.pth")

        # --- Improvement Check ---
        improved = False

        # Check AUC
        if roc_auc > best_auc:
            best_auc = roc_auc
            save_ckpt('bestAUC')
            print(f"--> New Best AUC! Saved model.")
            improved = True
            best_epoch_auc = epoch

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_ckpt('bestLoss')
            print(f"--> New Best Loss! Saved model.")
            # improved = True  # Optional: You can decide if loss improvement counts for patience
            best_epoch_loss = epoch

        wandb.log({"Test AUC": roc_auc, "Best AUC": best_auc, "Test Loss": avg_val_loss, "Epoch": epoch})
        print(f"Epoch: {epoch}, AUC: {roc_auc:.5f}, Loss: {avg_val_loss:.5f}")

        if improved:
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
            print(f"No improvement. Patience: {steps_since_improvement}/{args.patience}")

        if steps_since_improvement >= args.patience:
            print(f"Early stopping triggered at Epoch {epoch}")
            print("Best Epoch AUC: ", best_epoch_auc)
            print("Best Epoch Loss: ", best_epoch_loss)
            return  # Exit function completely

        cls_model.train()


if __name__ == '__main__':
    # gpu
    gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # Remove the first argument (GPU ID) from sys.argv so argparse doesn't see it
    if len(sys.argv) > 1: sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    # Add argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=str, default='0')

    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)

    # Model Params
    parser.add_argument('--cls_type', type=str, default='VIBCNN', help='CNN, VIBCNN, HO, VIBHO')
    parser.add_argument('--pooling', type=str, default='average', help='average, max')
    parser.add_argument('--depth', type=int, default=4, help='1-10')
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--free_bits', type=float, default=0.5, help='Free bits threshold for KL divergence')
    parser.add_argument('--kl', type=float, default=0.001, help='KL divergence weight for VIBCNN and VIBHO')

    parser.add_argument('--adaptive_kl', type=bool, default=False, help='If True, increase KL weight for smaller data proportions')
    parser.add_argument('--ramp_start', type=int, default=1000, help='Step to start KL annealing')
    parser.add_argument('--ramp_end', type=int, default=150000, help='Step to end KL annealing')

    parser.add_argument('--lpips', type=bool, default=False)
    parser.add_argument('--cycoptim', type=bool, default=False)
    parser.add_argument('--anneal', type=bool, default=False, help='If True, use KL annealing schedule')

    # Dataset Params
    parser.add_argument('--noise_level', type=float, default=35.0, help='noise level')
    parser.add_argument('--signal_location', type=str, default='ske', help='ske, sks')
    parser.add_argument('--amplitude', type=float, default=0.05, help='Signal amplitude')
    parser.add_argument('--sigma', type=float, default=3.0, help='Signal width')
    parser.add_argument('--blur_sigma', type=float, default=0.0, help='Blur width')

    # Others
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_parallel', type=bool, default=False)
    parser.add_argument('--use_ram', type=int, default=0)

    # Scheduler
    parser.add_argument('--scheduler_step_size', type=int, default=10)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--scheduler_patience', type=float, default=10)


    # Validation and Save Params
    # parser.add_argument('--val_interval', type=int, default=500, help='Validate every N iterations')
    parser.add_argument('--patience', type=int, default=30, help='Stop after N validation checks without improvement')

    # Paths and settings
    parser.add_argument('--srtype', type=str, default='UNet', help='UNet, UNet_Small, UNet_Tiny')
    parser.add_argument('--n_data', type=float, default=170.0, help='1~170')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train_type', type=str, default='measure', help='cls, measure, recon')

    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--train_image_base', type=str, default='/shared/anastasio-s2/SI/HCP_selected/background/train/')
    parser.add_argument('--val_image_base', type=str, default='/shared/anastasio-s2/SI/HCP_selected/background/val/')
    parser.add_argument('--test_image_base', type=str, default='/shared/anastasio-s2/SI/HCP_selected/background/test/')
    parser.add_argument('--save_model_base', type=str, default='checkpoints')

    args = parser.parse_args()

    if args.cls_type == 'VIBCNN':
        save_folder = 'VIBCE'
    else:
        # Need to change
        save_folder = 'CNNIO'

    if args.data is None:
        if args.blur_sigma > 0.0:
            args.data = f'{args.signal_location}_{args.sigma}_{args.amplitude}_{args.noise_level}_{args.blur_sigma}'
        else:
            args.data = f'{args.signal_location}_{args.sigma}_{args.amplitude}_{args.noise_level}'

    # ------ Wandb Setting ------ ##

    wandb.init(project='VIBCE-SKS-MRI-SMALL', config=args)  # Initialize here to capture config
    config = wandb.config

    # Override args with sweep values if they exist
    if hasattr(config, 'z_dim'):
        args.z_dim = config.z_dim
    if hasattr(config, 'kl'):
        args.kl = config.kl
    if hasattr(config, 'lr'):
        args.lr = config.lr
    if hasattr(config, 'batch_size'):
        args.batch_size = config.batch_size

    # Update run name to reflect sweep params
    # wandb.run.name = ('VIBCE_{}_{}_{}_{}_z{}_kl{}_lr{}_bsize{}'
    #                   .format(args.data[:3], args.train_type, args.cls_type, args.proporation,
    #                           args.z_dim, args.kl, args.lr, args.batch_size))

    args.param_setting = 'd{}_z{}_kl{}_lr{}_b{}'.format(args.depth, args.z_dim, args.kl, args.lr, args.batch_size)
    wandb.run.name = ('{}_{}_{}_{}_{}_{}'
                      .format(save_folder, args.data[:3], args.train_type, args.cls_type, args.n_data,
                              args.param_setting))

    total_images = args.n_data * 1000
    end_index = int((total_images - 1) // 2000)

    if end_index == 0:
        # If we only need the first file
        args.train_image_path = os.path.join(args.train_image_base, 'dataset-000000.tar')
    else:
        # If we need multiple files
        args.train_image_path = os.path.join(args.train_image_base, f'dataset-{{000000..{end_index:06d}}}.tar')

    # args.val_image_path = f'/shared/anastasio-s2/SI/HCP_selected/background/val/{args.data}/dataset-{{000000..000004}}.tar'

    args.val_image_path = os.path.join(args.val_image_base, f'{args.data}/dataset-{{000000..000004}}.tar')

    # n_workers
    num_files = end_index + 1
    args.num_workers = min(args.num_workers, num_files)

    # Adaptive KL weight based on data proportion
    if args.adaptive_kl:
        # Increase KL weight when data proportion is smaller
        # Base KL weight scaled inversely with proportion
        args.kl = args.kl / args.proporation
        print(f"Adaptive KL enabled. KL weight adjusted to: {args.kl}")

    save_base = 'checkpoints'

    from datetime import datetime

    current_date = datetime.now().strftime('%Y-%m-%d')
    args.save_model_path = os.path.join(args.save_model_base,
                                        '{}/{}/{}/{}'.format(args.data, args.n_data, save_folder, current_date))

    train(args)
