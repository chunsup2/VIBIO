import os
os.environ["WANDB__SERVICE_WAIT"] = "120"

import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from network.CNN_IO_new import BinaryClassifier

from torch.utils.data import DataLoader
import webdataset as wds

import numpy as np
import argparse
import wandb
import random
from datetime import datetime
from sklearn.metrics import roc_auc_score

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def preprocess_wds_sample(sample):
    """
    Parses the decoded WebDataset sample.
    sample[0] is 'image.npy' (numpy array)
    sample[1] is 'info.json' (dictionary containing 'label', 'io_lr', 'log_io_lr')
    """
    image_npy, info_json = sample

    # Convert numpy array to float32 tensor and add the channel dimension (1, H, W)
    img_tensor = torch.tensor(image_npy, dtype=torch.float32).unsqueeze(0)

    # Extract the binary label as a long tensor for CrossEntropyLoss
    label_tensor = torch.tensor(info_json['label'], dtype=torch.long)

    return img_tensor, label_tensor


def train(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    print(f"Using device: {device}")

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    # Initialize strictly the CNN-IO Binary Classifier
    cls_model = BinaryClassifier(args.depth, num_classes=2, input_height=64, input_width=64, pooling=args.pooling).to(device)

    if args.data_parallel and torch.cuda.device_count() > 1:
        cls_model = nn.DataParallel(cls_model)

    optimizer = optim.Adam(list(cls_model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.scheduler_gamma,
        patience=args.scheduler_patience, verbose=True
    )
    cls_criterion = nn.CrossEntropyLoss().to(device)

    # WebDataset Setup
    # total_train_images = args.num_train * 2  # H0 + H1
    total_train_images = int(args.n_data * 1000) * 2
    total_val_images = args.num_val * 2

    # Dynamically scale shuffle buffer if dataset is very small
    n_shuffle = total_train_images if total_train_images < 10000 else 10000

    # wds.decode() natively converts .npy to numpy arrays and .json to python dicts
    train_dataset = (
        wds.WebDataset(args.train_image_path, shardshuffle=True)
        .shuffle(n_shuffle, initial=n_shuffle)
        .decode("torch")
        .to_tuple("image.npy", "info.json")
        .map(preprocess_wds_sample)
        .slice(total_train_images)  # STRICTLY cut off any excess images from the final loaded shard
        .batched(args.batch_size)
        .with_length(total_train_images // args.batch_size)
    )

    val_dataset = (
        wds.WebDataset(args.val_image_path, shardshuffle=False)
        .decode("torch")
        .to_tuple("image.npy", "info.json")
        .map(preprocess_wds_sample)
        .batched(args.batch_size)
        .with_length(int(np.ceil(total_val_images / args.batch_size)))
    )

    train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=None,
                                  prefetch_factor=args.num_workers if args.num_workers > 0 else None)
    val_dataloader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=None,
                                prefetch_factor=args.num_workers if args.num_workers > 0 else None)

    best_auc = 0.5
    best_val_loss = float('inf')
    steps_since_improvement = 0

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        cls_model.train()

        # Training Loop
        for i, (images, labels) in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch}")):
            images = images.to(device)
            labels = labels.to(device)

            outputs = cls_model(images)
            loss = cls_criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log training metrics
        wandb.log({"Train Loss": loss.item(), "Epoch": epoch})

        # Validation Loop
        cls_model.eval()
        running_val_loss = 0.0
        val_batch_count = 0

        true_labels = []
        predicted_probs = []

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(val_dataloader, desc="Validating")):
                images = images.to(device)
                labels = labels.to(device)

                outputs = cls_model(images)
                batch_loss = cls_criterion(outputs, labels)

                running_val_loss += batch_loss.item()
                val_batch_count += 1

                # Calculate probabilities for AUC
                cls_probs = F.softmax(outputs, dim=1)

                predicted_probs.extend(cls_probs[:, 1].cpu().numpy())  # Store positive class prob
                true_labels.extend(labels.cpu().numpy())

        avg_val_loss = running_val_loss / val_batch_count if val_batch_count > 0 else float('inf')

        # Calculate AUC for Binary Classification
        roc_auc = roc_auc_score(true_labels, predicted_probs)
        if roc_auc < 0.5: roc_auc = 1.0 - roc_auc

        scheduler.step(roc_auc)

        # Improvement Checking & Model Saving
        state_dict = cls_model.module.state_dict() if hasattr(cls_model, 'module') else cls_model.state_dict()
        base_path = args.save_model_path
        fname_core = f"CNNIO_{args.param_setting}"

        def save_ckpt(suffix):
            torch.save(state_dict, os.path.join(base_path, f"{fname_core}_{suffix}.pth"))

        improved = False

        if roc_auc > best_auc:
            best_auc = roc_auc
            save_ckpt('bestAUC')
            print(f"--> New Best AUC: {best_auc:.5f}! Saved model.")
            improved = True
            best_epoch_auc = epoch

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_ckpt('bestLoss')
            print(f"--> New Best Loss: {best_val_loss:.5f}! Saved model.")
            best_epoch_loss = epoch

        wandb.log({
            "Test AUC": roc_auc,
            "Best AUC": best_auc,
            "Test Loss": avg_val_loss
        })

        print(f"Epoch {epoch} Results | AUC: {roc_auc:.5f} | Loss: {avg_val_loss:.5f}")

        # Early Stopping Logic
        if improved:
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
            print(f"No improvement. Patience: {steps_since_improvement}/{args.patience}")

        if steps_since_improvement >= args.patience:
            print(f"Early stopping triggered at Epoch {epoch}")
            print(f"Best Epoch AUC: {best_epoch_auc}")
            print(f"Best Epoch Loss: {best_epoch_loss}")
            break


if __name__ == '__main__':
    # gpu
    gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # Remove the first argument (GPU ID) from sys.argv so argparse doesn't see it
    if len(sys.argv) > 1: sys.argv = [sys.argv[0]] + sys.argv[2:]

    parser = argparse.ArgumentParser()

    # Define Data Size Argument
    parser.add_argument('--n_data', type=float, default=500.0,
                        help='Number of training images per class (in thousands)')

    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)

    # Model Params (Pruned to just what CNN-IO needs)
    parser.add_argument('--pooling', type=str, default='average', help='average, max')
    parser.add_argument('--depth', type=int, default=4, help='1-10')

    # Hardware / Processing
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_parallel', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Scheduler & Callbacks
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--scheduler_patience', type=float, default=10)
    parser.add_argument('--patience', type=int, default=15, help='Stop after N val checks without improvement')

    # Dataset Paths & Settings
    # Matches the output format from your data generation script
    parser.add_argument('--dataset_name', type=str, default='ske_bke_gauss_A0.2_std30_num500000_num50000',
                        help='Folder name of the dataset')
    parser.add_argument('--data_dir', type=str, default='/shared/anastasio-s2/SI/HCP_selected')
    parser.add_argument('--save_model_base', type=str, default='checkpoints')
    # parser.add_argument('--num_train', type=int, default=500000)
    parser.add_argument('--num_val', type=int, default=50000)

    args = parser.parse_args()

    # Total training images = (n_data * 1000) for H0 + (n_data * 1000) for H1
    total_train_images = int(args.n_data * 1000) * 2

    # Assuming shards contain exactly 10,000 items (based on your generation script)
    end_index = max(0, int((total_train_images - 1) // 10000))
    train_dir = os.path.join(args.data_dir, args.dataset_name, 'train')

    if end_index == 0:
        args.train_image_path = os.path.join(train_dir, 'dataset-000000.tar')
    else:
        args.train_image_path = os.path.join(train_dir, f'dataset-{{000000..{end_index:06d}}}.tar')

    num_train_files = end_index + 1
    args.num_workers = min(args.num_workers, num_train_files)

    # Construct WebDataset paths based on the new folder structure
    # args.train_image_path = os.path.join(args.data_dir, args.dataset_name, 'train', 'dataset-{000000..000099}.tar')
    args.val_image_path = os.path.join(args.data_dir, args.dataset_name, 'val', 'dataset-{000000..000009}.tar')

    args.param_setting = f'd{args.depth}_lr{args.lr}_b{args.batch_size}'

    # Setup Save Path
    current_date = datetime.now().strftime('%Y-%m-%d')
    # args.save_model_path = os.path.join(args.save_model_base, args.dataset_name, 'CNNIO', current_date)

    args.save_model_path = os.path.join(
        args.save_model_base,
        args.dataset_name,
        f'ndata_{args.n_data}k',
        'CNNIO',
        current_date
    )

    # Initialize wandb
    wandb.init(project='CNNIO-SKEBKE', config=args)
    wandb.run.name = f'CNNIO_{args.dataset_name[:15]}_n{args.n_data}k_{args.param_setting}'

    train(args)