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

# Model definitions consistent with training script
from network.UNet import UNet
from network.CNN_IO import BinaryClassifier
from network.ResNet_IO import ResNetX
from network.VAE import VIBCNN, VIBHO
from network.HO import SLNNHO

from dataloader import MRIDataset
from utils import load_model, update_results_csv

def compute_auc_with_flip(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute ROC AUC and apply flip if < 0.5 (to keep consistency with main script).
    """
    auc = roc_auc_score(y_true, scores)
    if auc < 0.5:
        auc = 1.0 - auc
    return auc



def bootstrap_auc(y_true: np.ndarray, scores: np.ndarray, n_boot: int = 1000, seed: int = 42):
    """
    Returns bootstrap mean AUC and std(AUC)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys = y_true[idx]
        ss = scores[idx]

        # Need both classes
        if ys.max() == ys.min():
            continue

        try:
            auc_b = compute_auc_with_flip(ys, ss)
            aucs.append(auc_b)
        except Exception:
            continue

    if len(aucs) == 0:
        return float('nan'), float('nan')

    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs, ddof=1))
    return auc_mean, auc_std


def _load_dp_checkpoint(model_dp, ckpt_path, device):
    # print(ckpt_path)
    raw_state = torch.load(ckpt_path, map_location=device)
    new_state = {}
    for k, v in raw_state.items():
        if not k.startswith('module.'):
            new_state['module.' + k] = v
        else:
            new_state[k] = v
    model_dp.load_state_dict(new_state)

def build_models(args, device):
    recon_model = None
    if args.train_type == 'recon':
        recon_model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
        recon_model = torch.nn.DataParallel(recon_model)
        ckpt = os.path.join(args.save_model_path, args.srtype, 'srnet.pth')
        _load_dp_checkpoint(recon_model, ckpt, device)
        recon_model.eval()

    num_classes = 3 if 'c3' in args.data else 2

    if args.cls_type == 'ResNet':
        base = ResNetX(args.depth, num_classes=num_classes)
    elif args.cls_type == 'CNN':
        base = BinaryClassifier(args.depth, num_classes=num_classes)
    elif args.cls_type == 'HO':
        base = SLNNHO()
    elif args.cls_type == 'VIBHO':
        base = VIBHO()
    elif args.cls_type == 'VIBCNN':
        base = VIBCNN(args.depth, args.z_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown cls_type: {args.cls_type}")

    cls_model = torch.nn.DataParallel(base.to(device))
    ckpt = os.path.join(
        args.save_model_path,
        args.srtype,
        f"ema{args.cls_type}_{args.train_type}_{args.kl}_{args.lr}_depth{args.depth}_z{args.z_dim}_io{args.ioloss}.pth"
        # f"ema{args.cls_type}_{args.train_type}_{args.kl}_{args.ema_beta}.pth"
    )
    _load_dp_checkpoint(cls_model, ckpt, device)
    cls_model.eval()

    return recon_model, cls_model

def normal_IO_test(mu: np.ndarray,
                   mu0_mean: np.ndarray,
                   s: np.ndarray,
                   Kinv: np.ndarray) -> np.ndarray:
    """
    Test phase: Compute test statistic λ = s^T K^{-1} (f - mu0_mean) for each test sample.

    Parameters
    ----------
    mu        : ndarray, shape (M, D)
        Feature vectors of test samples
    mu0_mean  : ndarray, shape (D,)
        No-signal mean estimated during training
    s         : ndarray, shape (D,)
        Signal vector from training phase
    Kinv      : ndarray, shape (D, D)
        Inverse covariance estimated during training

    Returns
    -------
    lambda_   : ndarray, shape (M,)
        Test statistic for each test sample
    """
    mu = np.asarray(mu, dtype=float)
    # Align mu0_mean dimensions
    assert mu.shape[1] == mu0_mean.shape[0] == s.shape[0] == Kinv.shape[0] == Kinv.shape[1]

    # Center the data
    F = mu - mu0_mean            # (M, D)

    # Compute λ_i = s^T K^{-1} f_i
    lin = 2 * np.einsum('i,ij,nj->n', s, Kinv, F)

    const = s @ Kinv @ s                      # scalar
    # 3) Complete statistic
    lambda_full = lin - const

    return lambda_full

import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Test script: compute AUC & log to W&B")
    parser.add_argument('--data', type=str,
                        default='sks_3_0.02_35_c2_num_signals',
                        help="dataset identifier")
    parser.add_argument('--proporation', type=float, default=1.0,
                        help="training proportion used")
    parser.add_argument('--srtype', type=str, default='UNet',
                        help="UNet variant")
    parser.add_argument('--cls_type', type=str, default='VIBCNN',
                        choices=['ResNet','CNN','HO','VIBHO','VIBCNN'],
                        help="classifier type")
    parser.add_argument('--train_type', type=str, default='cls',
                        choices=['cls','measure','recon'],
                        help="training mode")
    parser.add_argument('--depth', type=int, default=6,
                        help="network depth")
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--kl', type=float, default=0.003,
                        help='KL value used in training')
    parser.add_argument('--ema_beta', type=float, default=0.99)
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--free_bits', type=float, default=0.001, help='Free bits threshold for KL divergence')
    parser.add_argument('--z_dim', type=int, default=10)
    parser.add_argument('--ioloss', type=float, default=500, help='0.1 for VIBHO, 0.0005 for VIBCNN')
    args = parser.parse_args()

    args.save_model_path = os.path.join('checkpoints', args.data, str(args.proporation))

    args.test_image_path = os.path.join(
        '/shared/anastasio-s2/SI/HCP_selected',
        args.data, 'test', 'data.h5'
    )
    args.train_image_path = os.path.join(
        '/shared/anastasio-s2/SI/HCP_selected',
        args.data, 'train', 'data.h5'
    )

    device = torch.device(args.device)

    # wandb.init(
    #     project='VIB-MRI',
    #     name=f"Test_{args.proporation}_{args.cls_type}_{args.train_type}_{args.data}_{args.kl}_{args.ema_beta}",
    #     config=vars(args),
    #     settings=wandb.Settings(_service_wait=120)
    # )

    recon_model, cls_model = build_models(args, device)

    test_ds = MRIDataset(args.test_image_path, proportion=1.0, use_padding=True,
    target_size=(272, 320),
    pad_mode="constant",   # 如果担心黑边可以改成 "reflect"
    pad_value=0.0)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=8
    )

    mu_all_train = np.zeros((1, args.z_dim))
    label_all_train = np.zeros((1, 1))

    npy_path = os.path.join(args.save_model_path, args.srtype)
    
    mu0_mean = np.load(os.path.join(npy_path, f"{args.cls_type}_{args.train_type}_kl{args.kl}_lr{args.lr}_depth{args.depth}_z{args.z_dim}_io{args.ioloss}_mu_ema.npy"))
    s = np.load(os.path.join(npy_path, f"{args.cls_type}_{args.train_type}_kl{args.kl}_lr{args.lr}_depth{args.depth}_z{args.z_dim}_io{args.ioloss}_s_ema.npy"))
    Kinv = np.load(os.path.join(npy_path, f"{args.cls_type}_{args.train_type}_kl{args.kl}_lr{args.lr}_depth{args.depth}_z{args.z_dim}_io{args.ioloss}_Kinv_ema.npy"))

    # mu0_mean = np.load(os.path.join(npy_path, f"{args.cls_type}_{args.train_type}_kl{args.kl}_beta{args.ema_beta}_mu_ema.npy"))
    # s = np.load(os.path.join(npy_path, f"{args.cls_type}_{args.train_type}_kl{args.kl}_beta{args.ema_beta}_s_ema.npy"))
    # Kinv = np.load(os.path.join(npy_path, f"{args.cls_type}_{args.train_type}_kl{args.kl}_beta{args.ema_beta}_Kinv_ema.npy"))

    # file_path = os.path.join(npy_path,f"{args.cls_type}_{args.train_type}_kl{args.kl}_beta{args.ema_beta}_mu_ema.npy")
    # print("加载路径:", file_path)




    # mu0_mean = np.load(os.path.join(npy_path, f"ema{args.cls_type}_{args.train_type}_kl{args.kl}_mu.npy"))
    # s = np.load(os.path.join(npy_path, f"ema{args.cls_type}_{args.train_type}_kl{args.kl}_s.npy"))
    # Kinv = np.load(os.path.join(npy_path, f"ema{args.cls_type}_{args.train_type}_kl{args.kl}_Kinv.npy"))

    # visualize mu0_mean, s, Kinv
    # mu0_mean is a 1D array, s is a 1D array, Kinv is a 2D array
    # show mu0_mean and s in heatmap
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mu0_mean.reshape(1, -1), cmap='hot', aspect='auto')
    plt.colorbar()
    plt.title('mu0_mean Heatmap')
    plt.subplot(1, 2, 2)
    plt.imshow(s.reshape(1, -1), cmap='hot', aspect='auto')
    plt.colorbar()
    plt.title('s Heatmap')
    wandb.log({
        'mu0_mean Heatmap': wandb.Image(plt),
        's Heatmap': wandb.Image(plt)
    })
    plt.close()
   
    # show heatmap of Kinv
    plt.figure(figsize=(8, 6))
    plt.imshow(Kinv, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Kinv Heatmap')
    wandb.log({
        'Kinv Heatmap': wandb.Image(plt)
    })
    plt.close()

    all_true, all_preds, all_probs = [], [], []
    mu_all = np.zeros((1, args.z_dim))
    label_all = np.zeros((1, 1))

    with torch.no_grad():
        for i, (imgs, measures, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
            imgs, measures = imgs.to(device), measures.to(device)
            labels = labels.long().to(device)

            if args.train_type == 'recon':
                feats = recon_model(imgs)
            elif args.train_type == 'measure':
                feats = measures
            else:
                feats = imgs

            if args.cls_type in ['CNN','ResNet']:
                logits = cls_model(feats)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                probs_np = probs.cpu().numpy()
            elif args.cls_type in ['HO','VIBHO']:
                out = cls_model(feats)
                if args.cls_type == 'VIBHO':
                    _, mu, _ = out
                    stats = mu.squeeze()
                else:
                    stats = out.squeeze()
                preds = (stats > 0.5).long()
                probs_np = stats.cpu().numpy()
            else:  # VIBCNN
                t, mu, logvar, recon = cls_model(feats, mode='test')
                probs = F.softmax(t, dim=1)
                preds = torch.argmax(probs, dim=1)
                probs_np = probs.cpu().numpy()

                labels_ = labels.view(-1, 1)
                mu_all = np.concatenate((mu_all, mu.detach().cpu().numpy()), axis=0)
                label_all = np.concatenate((label_all, labels_.detach().cpu().numpy()), axis=0)

            all_true.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs_np)

    mu_all = mu_all[1:]
    label_all = label_all[1:]
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    probs = np.concatenate(all_probs, axis=0)

    acc = accuracy_score(y_true, y_pred)
    wandb.log({'Test Accuracy': acc})

    # Calculate AUC/ROC based on number of classes
    if args.cls_type == 'VIBCNN':
        lambda_ = normal_IO_test(mu_all, mu0_mean, s, Kinv)
        scores = lambda_

        # show scores distribution with labels
        plt.figure()
        plt.hist(lambda_[y_true == 0], bins=50, alpha=0.5, label='Class 0')
        plt.hist(lambda_[y_true == 1], bins=50, alpha=0.5, label='Class 1')
        plt.xlabel('Test Statistic')
        plt.ylabel('Frequency')
        plt.title('Test Statistic Distribution')
        plt.legend()
        wandb.log({
            'Test Statistic Distribution': wandb.Image(plt)
        })
        plt.close()

        aucio = roc_auc_score(y_true, scores)
        fpr, tpr, _ = roc_curve(y_true, scores)

        aucio, aucio_std = bootstrap_auc(y_true, scores, n_boot=1000, seed=42)



        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {aucio:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('Gaussian IO ROC Curve')
        plt.legend(loc='lower right')
        wandb.log({
            'Test AUC': aucio,
            'ROC Curve': wandb.Image(plt)
        })
        plt.close()

    if probs.ndim == 1:
        # 1D scores
        scores = probs
        auc = roc_auc_score(y_true, scores)
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
    elif probs.shape[1] == 2:
        # Binary classification: take positive class probability
        pos_prob = probs[:, 1]
        auc = roc_auc_score(y_true, pos_prob)
        fpr, tpr, _ = roc_curve(y_true, pos_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('Binary ROC Curve')
        plt.legend(loc='lower right')
    else:
        # Multi-class OVR
        n_cls = probs.shape[1]
        y_bin = label_binarize(y_true, classes=list(range(n_cls)))
        auc = roc_auc_score(y_bin, probs, multi_class='ovr', average='macro')
        plt.figure()
        for i in range(n_cls):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            plt.plot(fpr, tpr,
                     label=f'Class {i} (AUC={roc_auc_score(y_bin[:, i], probs[:, i]):.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('Multiclass ROC Curve')
        plt.legend(loc='lower right')

    wandb.log({
        'Test AUC': auc,
        'ROC Curve': wandb.Image(plt)
    })
    plt.close()

    report = classification_report(y_true, y_pred, digits=4)
    print(f"Test AUC     : {aucio:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)
    # auc = aucio
    # if auc < 0.5:
    #     auc = 1 - auc
    aucio = round(aucio, 6)
    # csv_path = "results/{}.csv".format(args.data)
    # update_results_csv(csv_path, args.cls_type, args.proporation, auc)

    cls_type_with_params1 = f"VIB-IO-lr{args.lr}-depth{args.depth}-kl{args.kl}-z{args.z_dim}-ioloss{args.ioloss}"
    csv_path = f"resultsCVSskebkeNewnet0102/{args.data}_p{args.proporation}_vibio.csv"
    update_results_csv(csv_path, cls_type_with_params1, args.proporation, aucio, aucio_std)

    # show recon if VIBCNN
    if args.cls_type == 'VIBCNN':
        wandb.log({
            'Recon Image': wandb.Image(recon[0].detach().cpu().numpy())
        })
    
    # wandb.finish()



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()