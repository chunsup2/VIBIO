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

from dataloader import MRIDataset1
from utils import load_model, update_results_csv

import numpy as np


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
    Returns bootstrap mean AUC and std (AUC)
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
    # if args.train_type == 'recon':
    #     recon_model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    #     recon_model = torch.nn.DataParallel(recon_model)
    #     ckpt = os.path.join(args.save_model_path, args.srtype, 'srnet.pth')
    #     _load_dp_checkpoint(recon_model, ckpt, device)
    #     recon_model.eval()

    num_classes = 3 if 'c3' in args.data else 2

    if args.cls_type == 'ResNet':
        base = ResNetX(args.depth, num_classes=num_classes)
    elif args.cls_type == 'CNN':
        base = BinaryClassifier(args.depth, num_classes=num_classes)
    # elif args.cls_type == 'HO':
    #     base = SLNNHO()
    # elif args.cls_type == 'VIBHO':
    #     base = VIBHO()
    elif args.cls_type == 'VIBCNN':
        base = VIBCNN(args.depth, args.z_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown cls_type: {args.cls_type}")

    cls_model = torch.nn.DataParallel(base.to(device))
    if args.cls_type in ['VIBCNN', 'VIBHO']:
        ckpt = os.path.join(
            args.save_model_path,
            args.srtype,
            f"{args.cls_type}_{args.train_type}_kl{args.kl}_lr{args.lr}_d{args.depth}_z{args.z_dim}.pth"
        )
    else:
        ckpt = os.path.join(
            args.save_model_path,
            args.srtype,
            f"{args.cls_type}_{args.train_type}_lr{args.lr}_d{args.depth}.pth"
        )
    _load_dp_checkpoint(cls_model, ckpt, device)
    cls_model.eval()

    return recon_model, cls_model


def normal_IO_train(mu: np.ndarray,
                    label: np.ndarray):
    """
    Training phase: Estimate detector parameters s and Kinv, as well as no-signal mean mu0_mean from labeled features mu.

    Parameters
    ----------
    mu    : ndarray, shape (N, D)
        Feature vectors of all training samples
    label : ndarray, shape (N,) or (N,1)
        Binary classification labels 0/1 for each sample

    Returns
    -------
    mu0_mean : ndarray, shape (D,)
        Mean of no-signal class
    s        : ndarray, shape (D,)
        Signal vector = mu1_mean - mu0_mean
    Kinv     : ndarray, shape (D,D)
        Inverse covariance of background (after centering)
    """
    mu    = np.asarray(mu, dtype=float)
    label = np.asarray(label).flatten().astype(int)

    # Align lengths
    n = min(mu.shape[0], label.shape[0])
    mu, label = mu[:n], label[:n]

    # Separate two classes
    mask0 = (label == 0)
    mask1 = (label == 1)
    mu0   = mu[mask0]   # (N0, D)
    mu1   = mu[mask1]   # (N1, D)

    # Means
    mu0_mean = mu0.mean(axis=0)   # (D,)
    mu1_mean = mu1.mean(axis=0)   # (D,)

    # Signal vector
    s = mu1_mean - mu0_mean       # (D,)

    # Center background samples and estimate covariance
    F0 = mu0 - mu0_mean           # (N0, D)
    # Or use all samples minus mu0_mean: F_all = mu - mu0_mean
    cov = np.cov(F0, rowvar=False)  # (D, D)
    Kinv = np.linalg.inv(cov)       # (D, D)

    return mu0_mean, s, Kinv


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


def main(args):
    args.save_model_path = os.path.join('checkpoint1212', args.data, str(args.proporation))
    args.test_image_path = os.path.join(
        '/shared/anastasio-s2/SI/HCP_selected',
        args.data, 'test', 'data.h5'
    )
    args.train_image_path = os.path.join(
        '/shared/anastasio-s2/SI/HCP_selected',
        args.data, 'train', 'data.h5'
    )

    device = torch.device(args.device)

    wandb.init(
        project='VIBCE-MRI',
        name=f"Test_{args.proporation}_{args.cls_type}_{args.train_type}_{args.data}",
        config=vars(args),
        settings=wandb.Settings(_service_wait=120)
    )

    recon_model, cls_model = build_models(args, device)

    test_ds =  MRIDataset1(args.test_image_path, proportion=1.0, use_padding=True,
    target_size=(272, 320),
    pad_mode="constant",   # Change to "reflect" if concerned about black edges
    pad_value=0.0)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=8
    )

    mu_all_train = np.zeros((1, args.z_dim))
    label_all_train = np.zeros((1, 1))

    if args.cls_type == 'VIBCNN':
        train_ds = MRIDataset1(args.train_image_path, args.proporation, use_padding=True,
    target_size=(272, 320),
    pad_mode="constant",   # Change to "reflect" if concerned about black edges
    pad_value=0.0)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=False, drop_last=False, num_workers=8
        )
        with torch.no_grad():
            for i, (imgs, measures, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
                imgs, measures = imgs.to(device), measures.to(device)
                labels = labels.long().to(device)

                if 'c2' in args.data:
                    label_one_hot = F.one_hot(labels, num_classes=2).float()
                elif 'c3' in args.data:
                    label_one_hot = F.one_hot(labels, num_classes=3).float()

                t, mu, logvar, recon = cls_model(imgs, label_one_hot, mode='test')

                labels_ = labels.view(-1, 1)
                mu_all_train = np.concatenate((mu_all_train, mu.detach().cpu().numpy()), axis=0)
                label_all_train = np.concatenate((label_all_train, labels_.detach().cpu().numpy()), axis=0)
                mu_all_train = mu_all_train[1:]
                label_all_train = label_all_train[1:]

        mu0_mean, s, Kinv = normal_IO_train(mu_all_train, label_all_train)


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
                preds = (stats > 0).long()
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
    # If accuracy < 0.5, the model likely learned inverted predictions
    if acc < 0.5:
        acc = 1 - acc
        # Also flip the predictions for consistent reporting
        y_pred = 1 - y_pred
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

        aucmrigau, auc_boot_std_gau = bootstrap_auc(y_true, scores, n_boot=args.bootstrap, seed=42)
        # pos_prob = scores
        # bad_mask = ~np.isfinite(pos_prob)
        # if bad_mask.any():
        #     print("警告：发现 NaN/Inf 分数数量 =", bad_mask.sum())
        #     # 简单处理：把这些样本去掉（标签和分数一起删）
        #     y_true = y_true[~bad_mask]
        #     pos_prob = pos_prob[~bad_mask]

        # # 4) 分成 H0/H1
        # scores_neg = pos_prob[y_true == 0]
        # scores_pos = pos_prob[y_true == 1]
        # print("H0(标签0) 个数:", len(scores_neg))
        # print("H1(标签1) 个数:", len(scores_pos))
        # # 5) 生成 LABROC 格式的文件
        # out_path = "roc_input_labroc_mrivibce.txt"
        # with open(out_path, "w", newline="\n") as f:
        #     f.write("LABROC\n")
        #     f.write("Large\n")

        #         # ====== 写 H0（负类） ======
        #     for v in scores_neg:
        #         f.write(f"{v:.6f}\n")

        #         # H0 结束标记
        #     f.write("*\n")

        #         # ====== 写 H1（正类） ======
        #     for v in scores_pos:
        #         f.write(f"{v:.6f}\n")

        #         # H1 结束标记（文件结束）
        #     f.write("*\n")

        # print("文件已写入:", out_path)

        auc = roc_auc_score(y_true, scores)
        # if auc < 0.5:
        #     auc = 1 - auc
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('Gaussian IO ROC Curve')
        plt.legend(loc='lower right')
        wandb.log({
            'Test Gaussian AUC': auc,
            'Gaussian ROC Curve': wandb.Image(plt)
        })
        plt.close()
        aucmrigau = round(aucmrigau, 6)
        # type_ = args.cls_type + 'gaussian'
        cls_type_with_params = f"VIB-CE-lr{args.lr}-depth{args.depth}-z{args.z_dim}-kl{args.kl}"
        csv_path = f"resultsCVSskebkeNewnet1215/{args.data}_p{args.proporation}_ce.csv"
        update_results_csv(csv_path, cls_type_with_params, args.proporation, aucmrigau, auc_boot_std_gau)

    
    if args.cls_type == 'CNN':

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

            # bad_mask = ~np.isfinite(pos_prob)
            # if bad_mask.any():
            #         print("警告：发现 NaN/Inf 分数数量 =", bad_mask.sum())
            #         # 简单处理：把这些样本去掉（标签和分数一起删）
            #         y_true = y_true[~bad_mask]
            #         pos_prob = pos_prob[~bad_mask]

            #     # 4) 分成 H0/H1
            # scores_neg = pos_prob[y_true == 0]
            # scores_pos = pos_prob[y_true == 1]

            # print("H0(标签0) 个数:", len(scores_neg))
            # print("H1(标签1) 个数:", len(scores_pos))

            #     # 5) 生成 LABROC 格式的文件
            # out_path = "roc_input_labroc_mricnn.txt"
            # with open(out_path, "w", newline="\n") as f:
            #     # header
            #         f.write("LABROC\n")
            #         f.write("Large\n")

            #         # ====== 写 H0（负类） ======
            #         for v in scores_neg:
            #             f.write(f"{v:.6f}\n")

            #         # H0 结束标记
            #         f.write("*\n")

            #         # ====== 写 H1（正类） ======
            #         for v in scores_pos:
            #             f.write(f"{v:.6f}\n")

            #         # H1 结束标记（文件结束）
            #         f.write("*\n")

            # print("文件已写入:", out_path)

            auc = roc_auc_score(y_true, pos_prob)
            fpr, tpr, _ = roc_curve(y_true, pos_prob)
            plt.figure()
            plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('Binary ROC Curve')
            plt.legend(loc='lower right')
            aucmri, auc_boot_std = bootstrap_auc(y_true, pos_prob, n_boot=args.bootstrap, seed=42)
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
            aucmri, auc_boot_std = bootstrap_auc(y_true, probs, n_boot=args.bootstrap, seed=42)

        # if auc < 0.5:
        #     auc = 1 - auc
            
        wandb.log({
            'Test AUC': aucmri,
            'ROC Curve': wandb.Image(plt)
        })
        plt.close()

        report = classification_report(y_true, y_pred, digits=4)
        print(f"Test AUC     : {aucmri:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(report)
        aucmri = round(aucmri, 6)
        cls_type_with_params = f"CNN-IO-lr{args.lr}-depth{args.depth}"
        csv_path = f"resultsCVSskebkeNewnet1215/{args.data}_p{args.proporation}_cnn.csv"
        update_results_csv(csv_path, cls_type_with_params, args.proporation, aucmri, auc_boot_std)

    
    if args.cls_type == 'VIBCNN':
        wandb.log({
            'Recon Image': wandb.Image(recon[0].detach().cpu().numpy())
        })


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description="Test script: compute AUC & log to W&B")
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--z_dim', type=int, default=10)
    parser.add_argument('--data', type=str,
                        default='ske_3_0.04_15_c2_num_signals',
                        help="dataset identifier")
    parser.add_argument('--proporation', type=float, default=1.0,
                        help="training proportion used")
    parser.add_argument('--srtype', type=str, default='UNet',
                        help="UNet variant")
    parser.add_argument('--cls_type', type=str, default='CNN',
                        choices=['ResNet','CNN','HO','VIBHO','VIBCNN'],
                        help="classifier type")
    parser.add_argument('--train_type', type=str, default='cls',
                        choices=['cls','measure','recon'],
                        help="training mode")
    parser.add_argument('--depth', type=int, default=6,
                        help="network depth")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--kl', type=float, default=0.1,
                        help="KL divergence weight for VIBCNN")
    parser.add_argument('--bootstrap', type=int, default=1000,
                        help="number of bootstrap resamples for AUC std")

    args = parser.parse_args()

    main(args)