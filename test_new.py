import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve, auc

# --- Import your network and dataloader modules ---
from network.UNet import UNet
from network.CNN_IO_new import BinaryClassifier
from network.VAE import VIBCNN, VIBHO, VIBCNN_backup
from network.HO import SLNNHO
from network.ResNet_IO import ResNetX
from dataloader import MRIDataset1


def get_args():
    parser = argparse.ArgumentParser(description="Test script for MRI Models")

    # Hardware
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    # Data Params
    parser.add_argument('--data', type=str, default='sks_3_0.04_15_c2_num_signals')
    parser.add_argument('--proporation', type=float, default=1.0)
    parser.add_argument('--test_image_path', type=str, default=None)

    # Model Params
    parser.add_argument('--cls_type', type=str, default='VIBCNN', help='CNN, VIBCNN, HO, VIBHO, ResNet')
    parser.add_argument('--train_type', type=str, default='cls', help='cls, measure, recon')
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--srtype', type=str, default='UNet')

    # Checkpoint loading
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--ckpt_suffix', type=str, default='bestAUC')

    # Training hyperparameters for filename reconstruction
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--kl', type=float, default=0.001)

    # Bootstrap Params
    parser.add_argument('--n_boot', type=int, default=1000, help='Number of bootstrap iterations')

    return parser.parse_args()


def compute_auc_with_flip(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute ROC AUC and apply flip if < 0.5.
    """
    try:
        auc_val = roc_auc_score(y_true, scores)
        if auc_val < 0.5:
            auc_val = 1.0 - auc_val
        return auc_val
    except ValueError:
        return float('nan')


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

        # Need both classes to compute AUC
        if len(np.unique(ys)) < 2:
            continue

        try:
            auc_b = compute_auc_with_flip(ys, ss)
            if not np.isnan(auc_b):
                aucs.append(auc_b)
        except Exception:
            continue

    if len(aucs) == 0:
        return float('nan'), float('nan')

    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs, ddof=1))
    return auc_mean, auc_std


def build_model(args):
    device = torch.device(args.device)
    num_classes = 3 if 'c3' in args.data else 2

    if args.cls_type == 'ResNet':
        model = ResNetX(args.depth, num_classes=num_classes)
    elif args.cls_type == 'CNN':
        model = BinaryClassifier(args.depth, num_classes=num_classes)
    elif args.cls_type == 'VIBCNN':
        model = VIBCNN_backup(args.depth, args.z_dim, num_classes=num_classes)
    elif args.cls_type == 'HO':
        model = SLNNHO()
    elif args.cls_type == 'VIBHO':
        model = VIBHO()
    elif args.train_type == 'recon':
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    else:
        raise ValueError(f"Unknown cls_type {args.cls_type}")

    return model.to(device)


def load_checkpoint(model, ckpt_path, device):
    print(f"Loading checkpoint from: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)

    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())

    if ckpt_keys[0].startswith('module.') and not model_keys[0].startswith('module.'):
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    return model


def plot_roc_curve(y_true, y_probs, save_path, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    # Check flip for plotting visual consistency
    if roc_auc < 0.5:
        roc_auc = 1 - roc_auc
        # Flip curve visual? usually just label is enough, or flip probs
        # Here we just report the flipped AUC in label

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()


def test(args):
    device = torch.device(args.device)

    # 1. Setup Paths
    if args.test_image_path is None:
        args.test_image_path = f'/shared/anastasio-s2/SI/HCP_selected/{args.data}/test/data.h5'

    if args.checkpoint_path is None:
        if args.cls_type == 'VIBCNN':
            save_folder = 'VIBCE'
        else:
            save_folder = 'CNNIO'

        fname_core = f"{args.cls_type}_{args.train_type}_kl{args.kl}_lr{args.lr}_d{args.depth}_z{args.z_dim}"
        save_base = 'checkpoints'
        args.checkpoint_path = os.path.join(save_base,
                                            f'{args.data}/{args.proporation}/{save_folder}/{fname_core}_{args.ckpt_suffix}.pth')

    # 2. Build & Load
    model = build_model(args)
    model = load_checkpoint(model, args.checkpoint_path, device)
    model.eval()

    # 3. Data Loader
    test_dataset = MRIDataset1(args.test_image_path, proportion=1.0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers)

    # 4. Inference
    all_preds = []
    all_probs = []
    all_labels = []

    print("Starting Inference...")
    with torch.no_grad():
        for image, measure, label in tqdm(test_loader):
            image, measure, label = image.to(device), measure.to(device), label.to(device)
            label = label.long()

            if args.train_type == 'measure':
                feats = measure
            else:
                feats = image

            if args.cls_type in ['CNN', 'ResNet']:
                output = model(feats)
                probs = F.softmax(output, dim=1)
                pos_probs = probs[:, 1]
                preds = torch.argmax(probs, dim=1)

            elif args.cls_type == 'VIBCNN':
                t, mu, logvar, recon = model(feats, mode='test')
                probs = F.softmax(t, dim=1)
                pos_probs = probs[:, 1]
                preds = torch.argmax(probs, dim=1)

            elif args.cls_type in ['HO', 'VIBHO']:
                if args.cls_type == 'VIBHO':
                    _, mu, _ = model(feats)
                    output = mu
                else:
                    output = model(feats)
                output = output.squeeze()
                pos_probs = output
                preds = (output > 0.5).long()

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(pos_probs.cpu().numpy())

    # 5. Metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    num_classes = 3 if 'c3' in args.data else 2

    print("\n" + "=" * 40)
    print("TEST RESULTS")
    print("=" * 40)

    # --- Standard Metrics ---
    acc = accuracy_score(y_true, y_pred)
    # Check for inversion based on accuracy
    if acc < 0.5:
        print("Note: Accuracy < 0.5, treating model as inverted for metric reporting.")
        acc = 1.0 - acc
        y_pred = 1 - y_pred  # Flip predictions for report

    print(f"Accuracy:      {acc:.5f}")

    # --- Bootstrap AUC ---
    # Only perform bootstrap if binary classification (or adapted for it)
    if num_classes == 2:
        print(f"Running Bootstrap AUC ({args.n_boot} iterations)...")
        auc_mean, auc_std = bootstrap_auc(y_true, y_probs, n_boot=args.n_boot)
        print(f"Bootstrap AUC: {auc_mean:.5f} ± {auc_std:.5f}")
    else:
        # For multiclass, basic OVR AUC
        auc_val = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        print(f"Multiclass AUC: {auc_val:.5f}")

    print("-" * 40)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("-" * 40)

    # 6. Save Plot
    output_dir = os.path.dirname(args.checkpoint_path)
    if not os.path.exists(output_dir):
        output_dir = '.'

    plot_filename = os.path.join(output_dir, f"ROC_{args.ckpt_suffix}_{args.data}.png")

    # Use the mean bootstrap AUC for the title if available
    if num_classes == 2:
        title = f"{args.cls_type} ROC (AUC={auc_mean:.4f}±{auc_std:.4f})"
    else:
        title = f"{args.cls_type} ROC"

    plot_roc_curve(y_true, y_probs, plot_filename, title=title)
    print(f"ROC Plot saved to: {plot_filename}")


if __name__ == '__main__':
    gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # Remove the first argument (GPU ID) from sys.argv so argparse doesn't see it
    if len(sys.argv) > 1: sys.argv = [sys.argv[0]] + sys.argv[2:]

    args = get_args()


    test(args)