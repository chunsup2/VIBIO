import os
import sys
import argparse
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve, auc

import warnings
warnings.filterwarnings("ignore")

# --- Import your network and dataloader modules ---
# Ensure these are in your python path
from network.UNet import UNet
from network.CNN_IO_new import BinaryClassifier
from network.VAE import VIBCNN, VIBHO, VIBCNN_backup
from network.HO import SLNNHO
from network.ResNet_IO import ResNetX
from dataloader import MRIDataset1


def get_args():
    parser = argparse.ArgumentParser(description="Batch Test script for MRI Models")

    # Hardware
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    # Data Params
    parser.add_argument('--data', type=str, default='sks_3_0.04_15_c2_num_signals')
    parser.add_argument('--test_image_path', type=str, default=None)

    # Checkpoint Directory (Target Folder)
    parser.add_argument('--ckpt_dir', type=str, required=True, help="Folder containing .pth files")

    # Bootstrap Params
    parser.add_argument('--n_boot', type=int, default=1000, help='Number of bootstrap iterations')

    return parser.parse_args()


def parse_model_params_from_filename(filename):
    """
    Parses parameters from filename like:
    "VIBCNN_measure_d4_z16_kl1e-08_lr0.001_b64_bestAUC.pth"
    Returns a dictionary of params.
    """
    params = {}

    # 1. Extract Model Type (Start of string)
    # Assumes format: [ModelType]_[TrainType]_...
    parts = filename.split('_')
    params['cls_type'] = parts[0]

    # 2. Extract Train Type (cls, measure, recon)
    # Usually the second item, but let's check known types
    if 'measure' in filename:
        params['train_type'] = 'measure'
    elif 'recon' in filename:
        params['train_type'] = 'recon'
    else:
        params['train_type'] = 'cls'

    # 3. Extract Hyperparams using Regex
    # Depth (d4)
    depth_match = re.search(r'_d(\d+)', filename)
    params['depth'] = int(depth_match.group(1)) if depth_match else 4

    # Z dim (z16)
    z_match = re.search(r'_z(\d+)', filename)
    params['z_dim'] = int(z_match.group(1)) if z_match else 16

    # KL (kl1e-08 or kl0.001)
    kl_match = re.search(r'_kl([\d\.e-]+)', filename)
    params['kl'] = float(kl_match.group(1)) if kl_match else 0.001

    # LR (lr0.001)
    lr_match = re.search(r'_lr([\d\.e-]+)', filename)
    params['lr'] = float(lr_match.group(1)) if lr_match else 0.0001

    return params


def compute_auc_with_flip(y_true, scores):
    try:
        auc_val = roc_auc_score(y_true, scores)
        if auc_val < 0.5:
            auc_val = 1.0 - auc_val
        return auc_val
    except ValueError:
        return float('nan')


def bootstrap_auc(y_true, scores, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys = y_true[idx]
        ss = scores[idx]
        if len(np.unique(ys)) < 2: continue
        try:
            auc_b = compute_auc_with_flip(ys, ss)
            if not np.isnan(auc_b): aucs.append(auc_b)
        except:
            continue

    if len(aucs) == 0: return float('nan'), float('nan')
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def build_model(args, params):
    device = torch.device(args.device)
    # Determine num_classes based on dataset name
    num_classes = 3 if 'c3' in args.data else 2

    cls_type = params['cls_type']
    train_type = params['train_type']
    depth = params['depth']
    z_dim = params['z_dim']

    if cls_type == 'ResNet':
        model = ResNetX(depth, num_classes=num_classes)
    elif cls_type == 'CNN':
        model = BinaryClassifier(depth, num_classes=num_classes)
    elif cls_type == 'VIBCNN':
        # VIBCNN_backup is what was used in the original script
        model = VIBCNN(depth, z_dim, num_classes=num_classes)
        # model = VIBCNN_backup(depth, z_dim, num_classes=num_classes)
    elif cls_type == 'HO':
        model = SLNNHO()
    elif cls_type == 'VIBHO':
        model = VIBHO()
    elif train_type == 'recon':
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    else:
        # Fallback or raise error
        print(f"Warning: Unknown cls_type {cls_type}, defaulting to BinaryClassifier")
        model = BinaryClassifier(depth, num_classes=num_classes)

    return model.to(device)


def evaluate_single_model(model, loader, device, params, args, ckpt_path):
    # ckpt_path is passed so we can use it to name the .txt file
    model_name = os.path.basename(ckpt_path)

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    cls_type = params.get('cls_type', 'VIBCNN')  # Default to VIBCNN if missing
    train_type = params.get('train_type', 'measure')

    with torch.no_grad():
        # Iterate over loader
        for image, measure, label in tqdm(loader, desc=f"Testing {model_name}", leave=False):
            image, measure, label = image.to(device), measure.to(device), label.to(device)
            label = label.long()

            # Determine input features
            if train_type == 'measure':
                feats = measure
            else:
                feats = image

            # --- Inference Logic ---
            if cls_type in ['CNN', 'ResNet']:
                output = model(feats)
                # Assuming output is logits for 2 classes
                probs_full = F.softmax(output, dim=1)
                pos_probs = probs_full[:, 1]  # Probability of class 1
                preds = torch.argmax(probs_full, dim=1)

            elif cls_type == 'VIBCNN':
                # VIBCNN returns tuple: (logits, mu, logvar, recon)
                t, mu, logvar, recon = model(feats, mode='test')
                probs_full = F.softmax(t, dim=1)
                pos_probs = probs_full[:, 1]
                preds = torch.argmax(probs_full, dim=1)

            elif cls_type in ['HO', 'VIBHO']:
                # These might output a single scalar (logit or prob)
                if cls_type == 'VIBHO':
                    _, mu, _ = model(feats)
                    output = mu  # Using latent mean as score? Adjust if needed.
                else:
                    output = model(feats)

                output = output.squeeze()
                # If output is raw logit, sigmoid it. If it's already prob, keep it.
                # Assuming it behaves like a score where higher = class 1
                pos_probs = torch.sigmoid(output)
                preds = (pos_probs > 0.5).long()

            else:
                # Default generic handle
                output = model(feats)
                probs_full = F.softmax(output, dim=1)
                pos_probs = probs_full[:, 1]
                preds = torch.argmax(probs_full, dim=1)

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(pos_probs.cpu().numpy())

    # --- Metrics Processing ---
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # 1. Calculate Accuracy and check for Flipping
    # Sometimes unsupervised/semi-supervised models flip labels (0->1, 1->0)
    acc = accuracy_score(y_true, y_pred)

    flip_detected = False
    if acc < 0.5:
        acc = 1.0 - acc
        flip_detected = True
        # Invert probs so ROC curve goes above diagonal
        y_probs = 1.0 - y_probs

        # 2. Compute ROC Curve (FPF, TPF)
    # pos_label=1 assumes class 1 is the positive class
    fpr, tpr, _ = roc_curve(y_true, y_probs, pos_label=1)

    # 3. Save FPF and TPF to .txt file
    # Format: 2 columns (FPF, TPF) separated by tab, matching your requested format
    # Filename example: "my_model.pth" -> "my_model_roc.txt"
    txt_save_path = ckpt_path.replace('.pth', '_roc.txt')

    # Just in case the replace didn't work (e.g. extension wasn't .pth)
    if txt_save_path == ckpt_path:
        txt_save_path = ckpt_path + '_roc.txt'

    try:
        with open(txt_save_path, 'w') as f:
            # Write data without header, matching the example file
            for f_val, t_val in zip(fpr, tpr):
                f.write(f"{f_val:.9f}\t{t_val:.9f}\n")
        print(f"Saved ROC text file to: {txt_save_path}")
    except Exception as e:
        print(f"Error saving ROC txt file: {e}")

    # 4. Final AUC Calculation
    num_classes = 3 if 'c3' in args.data else 2

    if num_classes == 2:
        # Use existing bootstrap function if available, otherwise simple roc_auc_score
        if 'bootstrap_auc' in globals():
            auc_mean, auc_std = bootstrap_auc(y_true, y_probs, n_boot=args.n_boot)
        else:
            auc_mean = roc_auc_score(y_true, y_probs)
            auc_std = 0.0
    else:
        # For multi-class, standard ROC curve saving is more complex (one vs all).
        # This code above saves the curve for class 1 vs others implicitly if labels are 0/1.
        auc_mean = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        auc_std = 0.0

    return {
        'Filename': model_name,
        'Accuracy': acc,
        'AUC_Mean': auc_mean,
        'AUC_Std': auc_std,
        'Params': params
    }


def main(args):
    device = torch.device(args.device)

    # 1. Setup Data Paths
    if args.test_image_path is None:
        args.test_image_path = f'/shared/anastasio-s2/SI/HCP_selected/{args.data}/val/data.h5'

    print(f"Data Source: {args.test_image_path}")
    print(f"Checkpoint Dir: {args.ckpt_dir}")

    # 2. Load Dataset ONCE
    # We load data once and reuse it for all models to save IO time
    print("Loading Test Dataset...")
    test_dataset = MRIDataset1(args.test_image_path, proportion=1.0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers)

    # 3. Find all .pth files
    if not os.path.isdir(args.ckpt_dir):
        raise ValueError(f"Directory not found: {args.ckpt_dir}")

    files = [f for f in os.listdir(args.ckpt_dir) if f.endswith('.pth')]
    if not files:
        print("No .pth files found in directory.")
        return

    # Sort files to have a consistent order
    files.sort()

    results_list = []

    # 4. Loop through files
    for f_name in files:
        f_path = os.path.join(args.ckpt_dir, f_name)

        # Parse params from filename
        try:
            params = parse_model_params_from_filename(f_name)
        except Exception as e:
            print(f"Skipping {f_name}: Could not parse params. Error: {e}")
            continue

        print(f"\nProcessing: {f_name}")
        print(f"   -> Detected: Type={params['cls_type']}, Z={params['z_dim']}, D={params['depth']}")

        # Build Model
        try:
            model = build_model(args, params)

            # Load Checkpoint
            state_dict = torch.load(f_path, map_location=device)

            # Handle DataParallel 'module.' prefix if present
            if list(state_dict.keys())[0].startswith('module.') and not list(model.state_dict().keys())[0].startswith(
                    'module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)

            # Run Test
            res = evaluate_single_model(model, test_loader, device, params, args, f_name)
            results_list.append(res)

            print(f"   -> Acc: {res['Accuracy']:.4f} | AUC: {res['AUC_Mean']:.4f} | AUC_std: {res['AUC_Std']:.4f}")

        except Exception as e:
            print(f"   -> Failed to process {f_name}. Error: {e}")
            import traceback
            traceback.print_exc()

    # 5. Save Summary
    if results_list:
        df = pd.DataFrame(results_list)

        # Flatten params dict into columns for easier sorting in CSV
        param_df = pd.json_normalize(df['Params'])
        df = pd.concat([df.drop(['Params'], axis=1), param_df], axis=1)

        save_path = os.path.join(args.ckpt_dir, 'test_summary_results.csv')
        df.sort_values(by='AUC_Mean', ascending=False, inplace=True)
        df.to_csv(save_path, index=False)

        print("\n" + "=" * 50)
        print(f"Processing Complete. Summary saved to:\n{save_path}")
        print("=" * 50)
        print(df[['Filename', 'Accuracy', 'AUC_Mean', 'AUC_Std', 'z_dim']].head().to_string())


if __name__ == '__main__':
    # Handle GPU ID manual parsing
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        gpu_id = sys.argv[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        sys.argv = [sys.argv[0]] + sys.argv[2:]

    args = get_args()
    main(args)