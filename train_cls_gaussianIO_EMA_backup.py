import os
os.environ["WANDB__SERVICE_WAIT"] = "120"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from network.UNet import UNet
from network.CNN_IO import BinaryClassifier
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
from wandb import Image

from utils import load_model, normal_IO_train_torch, normal_IO_test_torch, normal_IO_train_torch1, \
    normal_IO_train_torch2, GaussianIO  # Modified import
from test import normal_IO_train, normal_IO_test  # Modified import

from sklearn.metrics import roc_auc_score, RocCurveDisplay  # Modified import
import matplotlib.pyplot as plt
import random

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  # Fixed weight initialization to reduce variance


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
    kspace_noisy = (kspace + noise)

    # Perform batched 2D Inverse FFT
    img_recon = torch.fft.ifft2(kspace_noisy)
    return torch.real(img_recon)


def train(args):
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

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
        cls_model = BinaryClassifier(args.depth, num_classes=2).to(device)
    elif args.cls_type == 'VIBCNN':
        cls_model = VIBCNN(args.depth, args.z_dim, num_classes=2).to(device)
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

    if args.use_ram:
        if args.train_type == 'measure':
            train_dataset = MRIDataset2_2(args.train_image_path, args.proporation)
            val_dataset = MRIDataset2_2(args.val_image_path, proportion=1.0)
        else:
            train_dataset = MRIDataset2(args.train_image_path, args.proporation)
            val_dataset = MRIDataset2(args.val_image_path, proportion=1.0)

    else:
        if args.train_type == 'measure':
            # train_dataset = MRIDataset1_2(args.train_image_path, args.proporation)
            # val_dataset = MRIDataset1_2(args.val_image_path, proportion=1.0)
            train_dataset = MRIDataset3_2(args.train_image_path, args.proporation)
            val_dataset = MRIDataset3_2(args.val_image_path, proportion=0.5)
        else:
            train_dataset = MRIDataset1(args.train_image_path, args.proporation)
            val_dataset = MRIDataset1(args.val_image_path, proportion=1.0)

    if args.use_ram:
        with h5py.File(args.train_image_path, "r") as f:
            all_labels_train = f['label'][:train_dataset.selected_length]
    else:
        mapped_data = np.load(os.path.join(args.train_image_path, 'label.npy'))
        all_labels_train = mapped_data[:train_dataset.selected_length]

    train_class_counts = np.bincount(all_labels_train)
    train_weights = 1. / train_class_counts
    train_samples_weights = torch.from_numpy(train_weights[all_labels_train])
    train_sampler = WeightedRandomSampler(train_samples_weights, len(train_samples_weights))

    # training check
    # train_dataset = Subset(train_dataset, range(1000))
    # val_dataset = Subset(val_dataset, range(1000))

    train_dataloader = DataLoader(
        train_dataset,  # Your dataset
        batch_size=args.batch_size,  # Batch size (samples per batch)
        shuffle=False,  # Whether to shuffle data
        drop_last=True,  # Drop incomplete batches
        num_workers=args.num_workers,  # Number of CPU processes for data reading
        # pin_memory=True,  # Accelerate CPU->GPU copy
        # persistent_workers=True,  # Workers do not restart repeatedly
        # prefetch_factor=4,  # How many batches to prefetch
        sampler=train_sampler
    )

    val_dataloader = DataLoader(
        val_dataset,  # Your dataset
        batch_size=args.batch_size,  # Batch size (samples per batch)
        shuffle=False,  # Whether to shuffle data
        drop_last=True,  # Drop incomplete batches
        num_workers=args.num_workers,  # Number of CPU processes for data reading
        # pin_memory=True,  # Accelerate CPU->GPU copy
        # persistent_workers=True,  # Workers do not restart repeatedly
        # prefetch_factor=4  # How many batches to prefetch
    )


    ## --- Training State Variables ---
    global_step = 0
    best_auc = 0.5
    best_val_loss = float('inf')

    # Early stopping variables
    steps_since_improvement = 0

    # Placeholders for EMA variables (needed for validation)
    mu_ema = None
    Kinv_ema = None
    s_ema = None
    K_ema = None

    # Before the loop
    # gaussian_io = GaussianIO(z_dim=args.z_dim, beta=args.ema_beta).to(device)

    for epoch in tqdm(range(args.epochs)):
        cls_model.train()

        # mu_ema = np.zeros((1, args.z_dim))
        # Kinv_ema = np.zeros((args.z_dim, args.z_dim))
        # s_ema = np.zeros((1, args.z_dim))
        # K_ema = np.zeros((args.z_dim, args.z_dim))

        for i, data in enumerate(tqdm(train_dataloader)):
            # global_step += 1

            # image, measure, task_label = image.to(device), measure.to(device), task_label.to(device)

            if args.train_type == 'measure':
                measure = data[0].to(device)
                task_label = data[1].to(device)
                feats = measure
            elif args.train_type == 'cls':
                image = data[0].to(device)
                # measure = data[1].to(device)
                task_label = data[2].to(device)
                feats = image
            else:
                raise ValueError(f"Unknown train_type {args.train_type}")

            # Remove one-hot encoding; ensure task_label is LongTensor
            task_label = task_label.long()

            # loss = None
            # kl_loss = 0.0
            # cls_loss = 0.0

            # loss = torch.tensor(0.0).to(device)
            # kl_loss = torch.tensor(0.0).to(device)
            # cls_loss = torch.tensor(0.0).to(device)
            # ho_loss = torch.tensor(0.0).to(device)
            # recon_loss = torch.tensor(0.0).to(device)

            if args.cls_type in ['CNN', 'ResNet']:
                cls = cls_model(feats)
                loss = cls_criterion(cls, task_label)

            elif args.cls_type == 'VIBCNN':
                # one hot the task_label: 0 -> [0, 1], 1 -> [1, 0]
                label_one_hot = F.one_hot(task_label, num_classes=2).float()
                t, mu, logvar, recon = cls_model(feats, label=label_one_hot)

                kl_loss = compute_kl(mu, logvar, free_bits=args.free_bits,  # Can be set to 0.0 during ablation
                                     clamp_logvar=True,
                                     logvar_min=-20.0,
                                     logvar_max=20.0, )  # Todo

                mu0_mean, s, Kinv, K = normal_IO_train_torch1(mu, task_label, args.ema_beta)
                # mu0_mean, s, Kinv, K = gaussian_io(mu, task_label)

                lambda_full = normal_IO_test_torch(mu, mu0_mean, s, Kinv)
                # convert lambda_full to tensor and maximize the distance between the two classes
                lambda_0 = lambda_full[task_label == 0]
                lambda_1 = lambda_full[task_label == 1]

                cls_loss = -torch.mean(lambda_1) + torch.mean(lambda_0)
                cls_loss = -torch.log(-cls_loss + 1e-8)  # log to avoid negative values

                loss = args.ioloss * cls_loss + args.kl * kl_loss

                # Update EMA variables for validation usage
                mu_ema = mu0_mean
                Kinv_ema = Kinv
                s_ema = s
                K_ema = K
            else:
                raise ValueError("Model type not handled")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ============================================================
        # VALIDATION & EARLY STOPPING (Iteration-based)
        # ============================================================
        # if global_step % args.val_interval == 0:
            # Log training metrics
        # Todo
        wandb.log({
            "Train Loss": loss.item(),
            # "Train Recon Loss": recon_loss.item(),
            "Train KL Loss": args.kl * kl_loss.item(),
            # "Train HO Loss": ho_loss.item(),
            "Train Class Loss": args.ioloss * cls_loss.item(),
            "Epoch": epoch
        })

        # print(f"\n[Step {global_step}] Starting validation...")

        # Test
        cls_model.eval()
        running_val_loss = 0.0
        val_batch_count = 0

        mu_all_list = []
        label_all_list = []
        true_labels = []
        predicted_probs = []
        predicted_labels = []  # To store predicted class indices

        # mu_all = np.zeros((1, args.z_dim))
        # label_all = np.zeros((1, 1))

        with torch.no_grad():
            for i, data in enumerate(tqdm(val_dataloader)):
                # image, measure, task_label = image.to(device), measure.to(device), task_label.to(device)
                # measure, task_label = measure.to(device), task_label.to(device)

                if args.train_type == 'measure':
                    measure = data[0].to(device)
                    task_label = data[1].to(device)
                    feats = measure
                elif args.train_type == 'cls':
                    image = data[0].to(device)
                    # measure = data[1].to(device)
                    task_label = data[2].to(device)
                    feats = image
                else:
                    raise ValueError(f"Unknown train_type {args.train_type}")

                task_label = task_label.long()

                if args.cls_type in ['CNN', 'ResNet']:
                    cls = cls_model(feats)
                elif args.cls_type == 'VIBCNN':
                    t, mu, logvar, recon = cls_model(feats, mode='test')
                    cls = t

                    # labels_ = task_label.view(-1, 1)
                    mu_all_list.extend(mu.detach().cpu().numpy())
                    # label_all_list.append(task_label.cpu().numpy())

                    # mu_all = np.concatenate((mu_all, mu.detach().cpu().numpy()), axis=0)
                    # label_all = np.concatenate((label_all, labels_.detach().cpu().numpy()), axis=0)
                batch_loss = cls_criterion(cls, task_label)
                running_val_loss += batch_loss.item()
                val_batch_count += 1

                # if args.cls_type != 'VIBCNN':
                # For AUC calculation
                cls_probs = F.softmax(cls, dim=1)
                # predicted_probs.append(cls_probs.detach().cpu().numpy())
                predicted_probs.extend(cls_probs.detach().cpu().numpy())
                true_labels.extend(task_label.detach().cpu().numpy())

                # # Need it?
                # _, predicted = torch.max(cls_probs, 1)
                # predicted_labels.extend(predicted.detach().cpu().numpy())

                # probs_for_true_labels = cls_probs[torch.arange(cls_probs.size(0)), 0]
                # predicted_probs.extend(probs_for_true_labels.detach().cpu().numpy())

                # total += task_label.size(0)
                # correct += (predicted == task_label).sum().item()

                # if args.cls_type == 'HO' or args.cls_type == 'VIBHO':
                #     test_stat = cls.squeeze()
                #     predicted = (test_stat > 0.5).to(torch.int64)  # Using to(torch.int64) instead of long()
                #     predicted_labels.extend(predicted.detach().cpu().numpy())
                #     predicted_probs.extend(test_stat.detach().cpu().numpy())

                # if i == 0:
                #     image_ = image[0].squeeze(0).detach().cpu().numpy()
                #     measure_ = measure[0].squeeze(0).detach().cpu().numpy()
                #     wandb.log({
                #         # "Test Loss": loss.item(),
                #         "Images": [wandb.Image(image_, caption="Object"),
                #                    wandb.Image(measure_, caption="Measurement")]
                #     })
                #     if args.train_type == 'recon':
                #         recon_ = recon[0].squeeze(0).detach().cpu().numpy()
                #         wandb.log({
                #             "Images": [wandb.Image(recon_, caption="Reconstruction")]
                #         })

        # Calculate the accuracy
        # if args.cls_type == 'CNN' or args.cls_type == 'ResNet' or args.cls_type == 'VIBCNN':
        #     accuracy = 100 * correct / total
        #     wandb.log({"Test Accuracy": accuracy})

        # Convert true_labels and predicted_probs to appropriate format
        # if 'c3' in args.data:
        #     num_classes = 3
        # if 'c2' in args.data:
        #     num_classes = 2

        avg_val_loss = running_val_loss / val_batch_count if val_batch_count > 0 else float('inf')

        # One-hot encode true labels for ROC AUC calculation
        # true_labels_one_hot = np.eye(num_classes)[true_labels_array]

        if args.cls_type == 'VIBCNN':
            mu_all = np.array(mu_all_list)
            # mu_all = np.concatenate(mu_all_list, axis=0)
            # label_all = np.concatenate(label_all_list, axis=0)
            # mu_all = mu_all[1:]
            # label_all = label_all[1:]
            # to numpy
            mu_ema_ = mu_ema.detach().cpu().numpy()
            s_ema_ = s_ema.detach().cpu().numpy()
            Kinv_ema_ = Kinv_ema.detach().cpu().numpy()
            K_ema_ = K_ema.detach().cpu().numpy()

            # mu_ema_tensor = normal_IO_train_torch.mu0_ema
            # cov_ema_tensor = normal_IO_train_torch.cov_ema
            #
            # # Re-calculate s and Kinv from the stable EMA means for testing
            # mu1_ema_tensor = normal_IO_train_torch.mu1_ema
            #
            # s_ema_tensor = mu1_ema_tensor - mu_ema_tensor
            # Kinv_ema_tensor = torch.inverse(cov_ema_tensor)
            #
            # # Convert to numpy for the test function
            # mu_ema_ = mu_ema_tensor.detach().cpu().numpy()
            # s_ema_ = s_ema_tensor.detach().cpu().numpy()
            # Kinv_ema_ = Kinv_ema_tensor.detach().cpu().numpy()

            lambda_ = normal_IO_test(mu_all, mu_ema_, s_ema_, Kinv_ema_)

            true_labels_array = np.array(true_labels)
            roc_auc = roc_auc_score(true_labels_array, lambda_)

            predicted_probs_array = np.array(predicted_probs)
            pos_prob = predicted_probs_array[:, 1]
            roc_auc2 = roc_auc_score(true_labels_array, pos_prob)

            # show scores distribution with labels
            # fig = plt.figure()
            plt.figure()
            plt.hist(lambda_[true_labels_array == 0], bins=50, alpha=0.5, label='Class 0')
            plt.hist(lambda_[true_labels_array == 1], bins=50, alpha=0.5, label='Class 1')
            plt.xlabel('Test Statistic')
            plt.ylabel('Frequency')
            plt.title('Test Statistic Distribution')
            plt.legend()
            wandb.log({
                'Test Statistic Distribution': wandb.Image(plt)
            })
            plt.close()
            # plt.clf()
            # plt.close('all')


        else:
            true_labels_array = np.array(true_labels)
            predicted_probs_array = np.array(predicted_probs).reshape(-1, 1)
            # Calculate macro-average ROC AUC
            roc_auc = roc_auc_score(true_labels_array, predicted_probs_array[:, 1])

        if roc_auc < 0.5: roc_auc = 1 - roc_auc
        if roc_auc2 < 0.5: roc_auc2 = 1 - roc_auc2

        # Update scheduler
        scheduler.step(roc_auc)

        # --- Saving Helper ---
        state_dict = cls_model.module.state_dict() if hasattr(cls_model, 'module') else cls_model.state_dict()
        base_path = args.save_model_path
        fname_core = f"ema{args.cls_type}_{args.train_type}_{args.param_setting}"

        def save_ckpt(suffix):
            torch.save(state_dict, f"{base_path}/{fname_core}_{suffix}.pth")

            if args.cls_type == 'VIBCNN':
                np.save(f"{base_path}/{fname_core}_mu_ema_{suffix}.npy", mu_ema_)
                np.save(f"{base_path}/{fname_core}_s_ema_{suffix}.npy", s_ema_)
                np.save(f"{base_path}/{fname_core}_Kinv_ema_{suffix}.npy", Kinv_ema_)
                np.save(f"{base_path}/{fname_core}_K_ema_{suffix}.npy", K_ema_)

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

        wandb.log({"Test AUC": roc_auc, "Test AUC2": roc_auc2, "Best AUC": best_auc, "Test Loss": avg_val_loss, "Epoch": epoch})
        print(f"Epoch: {epoch}, AUC: {roc_auc:.5f}, Loss: {avg_val_loss:.5f}")


        if improved:
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
            print(f"No improvement. Patience: {steps_since_improvement}/{args.patience}")

        if steps_since_improvement >= args.patience:
            print(f"Early stopping triggered at Epoch {epoch})")
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

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Add argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=str, default='0')

    # Training Params
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)

    # Model Params
    parser.add_argument('--cls_type', type=str, default='VIBCNN', help='CNN, ResNet, HO, VIBHO')
    parser.add_argument('--depth', type=int, default=4, help='1-10')
    parser.add_argument('--free_bits', type=float, default=0.5, help='Free bits threshold for KL divergence')
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--kl', type=float, default=1, help='0.1 for VIBHO, 0.0005 for VIBCNN')
    parser.add_argument('--ioloss', type=float, default=1, help='0.1 for VIBHO, 0.0005 for VIBCNN')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--data_parallel', type=bool, default=False)
    parser.add_argument('--use_ram', type=int, default=0)

    parser.add_argument('--lpips', type=bool, default=False)
    parser.add_argument('--cycoptim', type=bool, default=False)

    # Scheduler
    parser.add_argument('--scheduler_step_size', type=int, default=10)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--scheduler_patience', type=float, default=10)

    # Validation and Save Params
    # parser.add_argument('--val_interval', type=int, default=500, help='Validate every N iterations')
    parser.add_argument('--patience', type=int, default=50, help='Stop after N validation checks without improvement')

    # Paths and settings
    parser.add_argument('--srtype', type=str, default='UNet', help='UNet, UNet_Small, UNet_Tiny')
    parser.add_argument('--proporation', type=float, default=1.0, help='0.001, 0.01, 0.1, 0.2, 0.5, 1.0')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train_type', type=str, default='measure', help='cls, measure, recon')

    parser.add_argument('--data', type=str, default='sks_3_0.04_15_c2_num_signals_noshuffle')
    parser.add_argument('--train_image_path', type=str, default=None)
    parser.add_argument('--test_image_path', type=str, default=None)
    parser.add_argument('--val_image_path', type=str, default=None)
    parser.add_argument('--save_model_path', type=str, default=None)
    parser.add_argument('--ema_beta', type=float, default=0.99)

    args = parser.parse_args()

    if args.cls_type == 'VIBCNN':
        save_folder = 'VIBIO'
    else:
        # Need to change
        save_folder = 'Unexpected'

    ## ------ Wandb Setting ------ ##

    wandb.init(project='VIBIO-SKS-MRI-FULL', config=args)  # Initialize here to capture config
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
    args.param_setting = 'd{}_z{}_kl{}_lr{}_io{}_b{}'.format(args.depth, args.z_dim, args.kl, args.lr,  args.ioloss, args.batch_size)
    wandb.run.name = ('{}_{}_{}_{}_{}_{}'
                      .format(save_folder, args.data[:3], args.train_type, args.cls_type, args.proporation,
                              args.param_setting))

    ## --------------------------- ##

    save_base = 'checkpoints'

    from datetime import datetime
    current_date = datetime.now().strftime('%Y-%m-%d')

    args.save_model_path = os.path.join(save_base, '{}/{}/{}/{}'.format(args.data, args.proporation, save_folder, current_date))

    if args.use_ram:
        args.train_image_path = '/shared/anastasio-s2/SI/HCP_selected/{}/train/data.h5'.format(args.data)
        args.test_image_path = '/shared/anastasio-s2/SI/HCP_selected/{}/test/data.h5'.format(args.data)
        args.val_image_path = '/shared/anastasio-s2/SI/HCP_selected/{}/val/data.h5'.format(args.data)
    else:
        args.train_image_path = '/shared/anastasio-s2/SI/HCP_selected/{}/train/'.format(args.data)
        args.test_image_path = '/shared/anastasio-s2/SI/HCP_selected/{}/test/'.format(args.data)
        args.val_image_path = '/shared/anastasio-s2/SI/HCP_selected/{}/val/'.format(args.data)

    # args.train_image_path = '/shared/anastasio-s2/SI/HCP_selected/{}/train/'.format(args.data)
    # args.test_image_path = '/shared/anastasio-s2/SI/HCP_selected/{}/test/'.format(args.data)
    # args.val_image_path = '/shared/anastasio-s2/SI/HCP_selected/{}/val/'.format(args.data)

    # args.train_image_path = '/home/chunsup2/data/SI/HCP_selected/{}/train/data.h5'.format(args.data)
    # args.test_image_path = '/home/chunsup2/data/SI/HCP_selected/{}/test/data.h5'.format(args.data)
    # args.val_image_path = '/home/chunsup2/data/SI/HCP_selected/{}/val/data.h5'.format(args.data)

    # args.train_image_path = '/scratch/chunsup2/HCP_selected/{}/train/data.h5'.format(args.data)
    # args.val_image_path = '/scratch/chunsup2/HCP_selected/{}/val/data.h5'.format(args.data)
    # args.test_image_path = '/scratch/chunsup2/HCP_selected/{}/test/data.h5'.format(args.data)

    train(args)
