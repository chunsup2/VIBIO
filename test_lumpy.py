#!/usr/bin/env python
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    roc_curve
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# Import lumpy-specific models
from network_lumpy.CNN_IO import BinaryClassifier
from network_lumpy.VAE import VIBCNN, VIBHO
from network_lumpy.HO import SLNNHO

from dataloader import LumpyDataset
from utils import update_results_csv

def _load_dp_checkpoint(model_dp, ckpt_path, device):
    """Load DataParallel checkpoint"""
    raw_state = torch.load(ckpt_path, map_location=device)
    new_state = {}
    for k, v in raw_state.items():
        if not k.startswith('module.'):
            new_state['module.' + k] = v
        else:
            new_state[k] = v
    model_dp.load_state_dict(new_state)

def build_lumpy_model(args, device):
    """Build lumpy-specific model"""
    num_classes = 2  # Lumpy is always binary classification
    
    if args.cls_type == 'CNN':
        base = BinaryClassifier(args.depth, num_classes=num_classes)
    elif args.cls_type == 'HO':
        base = SLNNHO()
    elif args.cls_type == 'VIBHO':
        base = VIBHO()
    elif args.cls_type == 'VIBCNN':
        base = VIBCNN(args.depth, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown cls_type: {args.cls_type}")

    cls_model = torch.nn.DataParallel(base.to(device))
    
    # Load checkpoint with KL parameter in filename for VIB models
    if args.cls_type in ['VIBHO', 'VIBCNN']:
        ckpt_path = os.path.join(args.save_model_path, f'clsnet_{args.cls_type}_kl{args.kl}_lr{args.lr}_kl{args.kl}.pth')
    else:
        ckpt_path = os.path.join(args.save_model_path, f'clsnet_{args.cls_type}_lr{args.lr}.pth')
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    _load_dp_checkpoint(cls_model, ckpt_path, device)
    cls_model.eval()
    
    return cls_model

def normal_IO_train(mu: np.ndarray, label: np.ndarray):
    """
    Train Gaussian IO detector from feature vectors and labels
    
    Parameters
    ----------
    mu : ndarray, shape (N, D)
        Feature vectors from all training samples
    label : ndarray, shape (N,)
        Binary labels (0/1) for each sample
    
    Returns
    -------
    mu0_mean : ndarray, shape (D,)
        Mean of background class
    s : ndarray, shape (D,)
        Signal vector = mu1_mean - mu0_mean
    Kinv : ndarray, shape (D, D)
        Inverse covariance matrix
    """
    mu = np.asarray(mu, dtype=float)
    label = np.asarray(label).flatten().astype(int)
    
    # Align lengths
    n = min(mu.shape[0], label.shape[0])
    mu, label = mu[:n], label[:n]
    
    # Separate classes
    mask0 = (label == 0)
    mask1 = (label == 1)
    mu0 = mu[mask0]  # Background samples
    mu1 = mu[mask1]  # Signal samples
    
    # Compute means
    mu0_mean = mu0.mean(axis=0)
    mu1_mean = mu1.mean(axis=0)
    
    # Signal vector
    s = mu1_mean - mu0_mean
    
    # Estimate covariance from centered background samples
    F0 = mu0 - mu0_mean
    cov = np.cov(F0, rowvar=False)
    Kinv = np.linalg.inv(cov)
    
    return mu0_mean, s, Kinv

def normal_IO_test(mu: np.ndarray, mu0_mean: np.ndarray, s: np.ndarray, Kinv: np.ndarray):
    """
    Test Gaussian IO detector on new samples
    
    Parameters
    ----------
    mu : ndarray, shape (M, D)
        Feature vectors from test samples
    mu0_mean : ndarray, shape (D,)
        Background mean from training
    s : ndarray, shape (D,)
        Signal vector from training
    Kinv : ndarray, shape (D, D)
        Inverse covariance from training
    
    Returns
    -------
    lambda_ : ndarray, shape (M,)
        Test statistics for each sample
    """
    mu = np.asarray(mu, dtype=float)
    assert mu.shape[1] == mu0_mean.shape[0] == s.shape[0] == Kinv.shape[0] == Kinv.shape[1]
    
    # Center features
    F = mu - mu0_mean
    
    # Compute test statistics
    lin = 2 * np.einsum('i,ij,nj->n', s, Kinv, F)
    const = s @ Kinv @ s
    lambda_full = lin - const
    
    return lambda_full

def main():
    parser = argparse.ArgumentParser(description="Test lumpy dataset models")
    parser.add_argument('--lr', type=float, default=1.0,
                        help="training proportion used")
    parser.add_argument('--data', type=str, 
                        default='lumpy_background_A0.2_std30_num200000',
                        help="lumpy dataset identifier")
    parser.add_argument('--proporation', type=float, default=1.0,
                        help="training proportion used")
    parser.add_argument('--cls_type', type=str, default='VIBHO',
                        choices=['CNN', 'HO', 'VIBHO', 'VIBCNN'],
                        help="classifier type")
    parser.add_argument('--depth', type=int, default=6,
                        help="network depth")
    parser.add_argument('--kl', type=float, default=0.001,
                        help="KL weight used in training (for path resolution)")
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Set paths - include depth for CNN and VIBCNN models, but not for HO and VIBHO (single-layer networks)
    if args.cls_type in ['CNN', 'VIBCNN']:
        args.save_model_path = os.path.join('checkpoint', args.data, str(args.proporation), f'depth_{args.depth}')
    else:  # HO and VIBHO don't use depth
        args.save_model_path = os.path.join('checkpoint', args.data, str(args.proporation))
    
    # args.test_image_path = os.path.join(
    #     '/shared/anastasio-s2/SI/HCP_selected', 
    #     args.data, 'test', 'data.h5'
    # )
    args.test_image_path = os.path.join(
        '/shared/anastasio-s2/SI/HCP_selected', 
        args.data, 'test', 'data.h5'
    )
    args.train_image_path = os.path.join(
        '/shared/anastasio-s2/SI/HCP_selected', 
        args.data, 'train', 'data.h5'
    )
    
    device = torch.device(args.device)
    
    # Initialize wandb
    wandb.init(
        project='VIB-Lumpy',
        name=f"Test_{args.proporation}_{args.cls_type}_{args.data}",
        config=vars(args),
        settings=wandb.Settings(_service_wait=120)
    )
    
    # Build model
    cls_model = build_lumpy_model(args, device)
    
    # Load test dataset
    test_ds = LumpyDataset(args.test_image_path, proportion=1.0)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=0
    )
    
    # For VIBCNN, we need to train Gaussian IO detector
    mu_all_train = np.zeros((1, 10))
    label_all_train = np.zeros((1, 1))
    
    if args.cls_type == 'VIBCNN':
        print("Training Gaussian IO detector for VIBCNN...")
        train_ds = LumpyDataset(args.train_image_path, proportion=args.proporation)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=False, drop_last=False, num_workers=4
        )
        
        with torch.no_grad():
            for i, (image, task_label) in tqdm(enumerate(train_loader), total=len(train_loader)):
                image = image.to(device)
                task_label = task_label.long().to(device)
                
                # One-hot encode labels
                label_one_hot = F.one_hot(task_label, num_classes=2).float()
                
                # Get latent features
                t, mu, logvar, recon = cls_model(image, label_one_hot, mode='train')
                
                # Collect features and labels
                labels_ = task_label.view(-1, 1)
                mu_all_train = np.concatenate((mu_all_train, mu.detach().cpu().numpy()), axis=0)
                label_all_train = np.concatenate((label_all_train, labels_.detach().cpu().numpy()), axis=0)
                
        mu_all_train = mu_all_train[1:]
        label_all_train = label_all_train[1:]
        
        # Train Gaussian IO detector
        mu0_mean, s, Kinv = normal_IO_train(mu_all_train, label_all_train)
        print(f"Gaussian IO detector trained with {len(mu_all_train)} samples")
    
    # Test loop
    print("Testing model...")
    all_true, all_preds, all_probs = [], [], []
    mu_all_test = np.zeros((1, 10))
    label_all_test = np.zeros((1, 1))
    
    with torch.no_grad():
        for i, (image, task_label) in tqdm(enumerate(test_loader), total=len(test_loader)):
            image = image.to(device)
            task_label = task_label.long().to(device)
            
            if args.cls_type in ['CNN']:
                logits = cls_model(image)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                probs_np = probs.cpu().numpy()
                
            elif args.cls_type in ['HO', 'VIBHO']:
                if args.cls_type == 'HO':
                    stats = cls_model(image).squeeze()
                else:  # VIBHO
                    t, mu, logvar = cls_model(image)
                    stats = t.squeeze()
                
                preds = (stats > 0.5).long()
                probs_np = stats.cpu().numpy()
                
            else:  # VIBCNN
                t, mu, logvar, recon = cls_model(image, mode='test')
                probs = F.softmax(t, dim=1)
                preds = torch.argmax(probs, dim=1)
                probs_np = probs.cpu().numpy()
                
                # Collect latent features for Gaussian IO
                labels_ = task_label.view(-1, 1)
                mu_all_test = np.concatenate((mu_all_test, mu.detach().cpu().numpy()), axis=0)
                label_all_test = np.concatenate((label_all_test, labels_.detach().cpu().numpy()), axis=0)
            
            all_true.append(task_label.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs_np)
    
    # Process results
    mu_all_test = mu_all_test[1:]
    label_all_test = label_all_test[1:]
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    probs = np.concatenate(all_probs, axis=0)
    
    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)
    wandb.log({'Test Accuracy': acc})
    
    # Calculate AUC and generate plots
    if args.cls_type == 'VIBCNN':
        # Use Gaussian IO detector
        lambda_ = normal_IO_test(mu_all_test, mu0_mean, s, Kinv)
        
        # Plot test statistic distribution
        plt.figure(figsize=(10, 6))
        plt.hist(lambda_[y_true == 0], bins=50, alpha=0.5, label='Background', color='blue')
        plt.hist(lambda_[y_true == 1], bins=50, alpha=0.5, label='Signal', color='red')
        plt.xlabel('Test Statistic')
        plt.ylabel('Frequency')
        plt.title('Gaussian IO Test Statistic Distribution')
        plt.legend()
        wandb.log({'Test Statistic Distribution': wandb.Image(plt)})
        plt.close()
        
        # Calculate AUC for Gaussian IO
        auc_gaussian = roc_auc_score(y_true, lambda_)
        if auc_gaussian < 0.5:
            auc_gaussian = 1 - auc_gaussian
        
        # Plot ROC curve for Gaussian IO
        fpr, tpr, _ = roc_curve(y_true, lambda_)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Gaussian IO AUC = {auc_gaussian:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Gaussian IO ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        wandb.log({'Gaussian IO ROC Curve': wandb.Image(plt)})
        plt.close()
        
        # Save Gaussian IO results
        auc_gaussian = round(auc_gaussian, 4)
        csv_path = f"results/{args.data}_p{args.proporation}_lr{args.lr}_kl{args.kl}_vibce.csv"
        # csv_path = f"results/{args.data}.csv"
        update_results_csv(csv_path, f"{args.cls_type}_CE", args.proporation, auc_gaussian)
        
        wandb.log({'Gaussian IO Test AUC': auc_gaussian})
        print(f"Gaussian IO Test AUC: {auc_gaussian:.4f}")
    
    # Calculate standard AUC
    if probs.ndim == 1:
        # 1D scores (HO, VIBHO)
        auc = roc_auc_score(y_true, probs)
        fpr, tpr, _ = roc_curve(y_true, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{args.cls_type} ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        
    else:
        # 2D probabilities (CNN, VIBCNN)
        pos_prob = probs[:, 1]  # Positive class probability
        auc = roc_auc_score(y_true, pos_prob)
        fpr, tpr, _ = roc_curve(y_true, pos_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{args.cls_type} ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
    
    wandb.log({'ROC Curve': wandb.Image(plt)})
    plt.close()
    
    # Handle AUC correction
    if auc < 0.5:
        auc = 1 - auc
    auc = round(auc, 4)
    
    # Save results to CSV
    # csv_path = f"results/{args.data}.csv"
    csv_path = f"results/{args.data}_p{args.proporation}_lr{args.lr}_cnn.csv"
    
    if not os.path.exists("results"):
        os.makedirs("results")
    
    cls_type_name = args.cls_type
    if args.cls_type in ['VIBHO', 'VIBCNN']:
        cls_type_name += f"_kl{args.kl}"
    
    update_results_csv(csv_path, cls_type_name, args.proporation, auc)
    
    # Log final results
    wandb.log({'Test AUC': auc})
    
    # Print summary
    print(f"\n=== Test Results Summary ===")
    print(f"Dataset: {args.data}")
    print(f"Model: {args.cls_type}")
    print(f"Proportion: {args.proporation}")
    if args.cls_type in ['VIBHO', 'VIBCNN']:
        print(f"KL Weight: {args.kl}")
    print(f"Test AUC: {auc:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    
    # Print classification report
    report = classification_report(y_true, y_pred, digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Show sample reconstruction for VIBCNN
    if args.cls_type == 'VIBCNN':
        with torch.no_grad():
            sample_image = test_ds[0][0].unsqueeze(0).to(device)
            sample_label = torch.tensor([0]).long().to(device)
            label_one_hot = F.one_hot(sample_label, num_classes=2).float()
            _, _, _, recon = cls_model(sample_image, label_one_hot, mode='test')
            
            wandb.log({
                'Sample Original': wandb.Image(sample_image[0].cpu().numpy()),
                'Sample Reconstruction': wandb.Image(recon[0].cpu().numpy())
            })

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Match train_cls_lumpy.py
    main()