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
    parser.add_argument('--num_workers', type=int, default=4)

    # Model Params
    parser.add_argument('--pooling', type=str, default='average', help='average, max')

    # Data Params
    parser.add_argument('--data', type=str, default='sks_3.0_0.2_25.0_c2_num_signals_diffusion_n')
    parser.add_argument('--test_image_path', type=str, default=None)

    # Checkpoint Directory (Target Folder)
    parser.add_argument('--ckpt_dir', type=str, required=True, help="Folder containing .pth files")

    # Bootstrap Params
    parser.add_argument('--n_boot', type=int, default=1000, help='Number of bootstrap iterations')

    return parser.parse_args()


def parse_model_params(filename):
    """
    Parses parameters (z_dim, depth, etc.) from filename using Regex.
    Target format: "emaVIBCNN_measure_kl1e-05_lr0.0001_d4_z256_io1.0_bestAUC.pth"
    """
    params = {}

    pattern = r"(?:ema)?([A-Za-z0-9]+)_([a-z]+)_d(\d+)_z(\d+)_kl([\de\.-]+)_lr([\de\.-]+)_io([\de\.-]+)"
    match = re.search(pattern, filename)

    if match:
        params['cls_type'] = match.group(1)
        params['train_type'] = match.group(2)
        params['depth'] = int(match.group(3))
        params['z_dim'] = int(match.group(4))
        params['kl'] = float(match.group(5))
        params['lr'] = float(match.group(6))
        params['ioloss'] = float(match.group(7))
    else:
        print(f"Warning: Regex did not match full pattern for {filename}. Using default fallback.")
        params['cls_type'] = 'VIBCNN'
        params['train_type'] = 'measure'
        params['depth'] = 6
        params['z_dim'] = 16
        params['ioloss'] = 1.0

        if '_z' in filename:
            try:
                z_part = filename.split('_z')[1].split('_')[0]
                params['z_dim'] = int(z_part)
            except:
                pass

    return params


def load_ema_stats(ckpt_path):
    """
    Constructs the NPY filenames based on the .pth filename.
    """
    if ckpt_path.endswith('_bestAUC.pth'):
        mu_path = ckpt_path.replace('_bestAUC.pth', '_mu_ema_bestAUC.npy')
        s_path = ckpt_path.replace('_bestAUC.pth', '_s_ema_bestAUC.npy')
        k_path = ckpt_path.replace('_bestAUC.pth', '_K_ema_bestAUC.npy')
    elif ckpt_path.endswith('_bestLoss.pth'):
        mu_path = ckpt_path.replace('_bestLoss.pth', '_mu_ema_bestLoss.npy')
        s_path = ckpt_path.replace('_bestLoss.pth', '_s_ema_bestLoss.npy')
        k_path = ckpt_path.replace('_bestLoss.pth', '_K_ema_bestLoss.npy')
    else:
        mu_path = ckpt_path.replace('.pth', '_mu_ema.npy')
        s_path = ckpt_path.replace('.pth', '_s_ema.npy')
        k_path = ckpt_path.replace('.pth', '_K_ema.npy')

    if os.path.exists(mu_path):
        return np.load(mu_path), np.load(s_path), np.load(k_path)
    else:
        print(f"   [!] EMA file not found: {os.path.basename(mu_path)}")
        return None, None, None


def normal_IO_test(mu, mu0_mean, s, Kinv):
    mu = np.asarray(mu, dtype=float)
    F = mu - mu0_mean
    lin = 2 * np.einsum('i,ij,nj->n', s, Kinv, F)
    const = s @ Kinv @ s
    lambda_full = lin - const
    return lambda_full


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
    num_classes = 3 if 'c3' in args.data else 2

    cls_type = params.get('cls_type', 'VIBCNN')
    depth = params.get('depth', 4)
    z_dim = params.get('z_dim', 16)

    if cls_type == 'ResNet':
        model = ResNetX(depth, num_classes=num_classes)
    elif cls_type == 'CNN':
        model = BinaryClassifier(depth, num_classes=num_classes, pooling=args.pooling)
    elif cls_type == 'VIBCNN':
        model = VIBCNN(depth, z_dim, num_classes=num_classes, pooling=args.pooling)
    elif cls_type == 'HO':
        model = SLNNHO()
    elif cls_type == 'VIBHO':
        model = VIBHO()
    elif params.get('train_type') == 'recon':
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    else:
        model = VIBCNN_backup(depth, z_dim, num_classes=num_classes)

    return model.to(device)


def evaluate_single_model(model, loader, device, params, args, ckpt_path):
    model.eval()

    mu0_mean, s, Kinv = None, None, None

    if params['cls_type'] == 'VIBCNN':
        mu0_mean, s, Kinv = load_ema_stats(ckpt_path)
        if mu0_mean is None:
            return None

    all_preds, all_scores, all_labels = [], [], []
    mu_list = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            feats = data[0].to('cuda')
            feats = feats.unsqueeze(1)
            label = torch.tensor([int(lbl) for lbl in data[1]]).to('cuda')

            if params['cls_type'] == 'VIBCNN':
                t, mu, logvar, recon = model(feats, mode='test')
                probs = F.softmax(t, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                mu_list.append(mu.cpu().numpy())
            else:
                output = model(feats)
                probs = F.softmax(output, dim=1)
                pos_probs = probs[:, 1]
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_scores.extend(pos_probs.cpu().numpy())

            all_labels.extend(label.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    if params['cls_type'] == 'VIBCNN':
        mu_all = np.concatenate(mu_list, axis=0)
        y_scores = normal_IO_test(mu_all, mu0_mean, s, Kinv)
    else:
        y_scores = np.array(all_scores)

    # ---------------------------------------------------------
    # AUTOMATIC LABROC TXT FILE GENERATOR
    # ---------------------------------------------------------
    # Filter the predicted scores into two separate arrays
    scores_h0 = y_scores[y_true == 0]  # Healthy
    scores_h1 = y_scores[y_true == 1]  # Diseased

    # Create a unique filename based on the current model's .pth name
    labroc_filename = ckpt_path.replace('.pth', '_labroc.txt')

    with open(labroc_filename, 'w') as fid:
        fid.write('LABROC\n')
        fid.write('Large\n')
        for score in scores_h0:
            fid.write(f'{score:.6f}\n')
        fid.write('*\n')
        for score in scores_h1:
            fid.write(f'{score:.6f}\n')
        fid.write('*')
    # ---------------------------------------------------------

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    if acc < 0.5:
        acc = 1.0 - acc

    num_classes = 3 if 'c3' in args.data else 2
    if num_classes == 2:
        auc_mean, auc_std = bootstrap_auc(y_true, y_scores, n_boot=args.n_boot)
    else:
        auc_mean = 0.0
        auc_std = 0.0

    return {
        'Filename': os.path.basename(ckpt_path),
        'Accuracy': acc,
        'AUC_Mean': auc_mean,
        'AUC_Std': auc_std,
        'z_dim': params.get('z_dim'),
        'depth': params.get('depth'),
        'io': params.get('ioloss')
    }


def main(args):
    device = torch.device(args.device)

    # 1. Setup Data
    if args.test_image_path is None:
        args.test_image_path = f'/shared/anastasio-s2/SI/HCP_selected/background/test/{args.data}/dataset-{{000000..000004}}.tar'
    print(f"Data: {args.test_image_path}")
    print(f"Ckpt Dir: {args.ckpt_dir}")

    print("Loading Test Dataset...")
    import webdataset as wds

    test_dataset = (
        wds.WebDataset(args.test_image_path, shardshuffle=False)
        .decode("torch")
        .to_tuple("npy", "cls")
        .batched(100)
        .with_length(100)
    )

    test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=None)

    # 2. Find Files
    if not os.path.exists(args.ckpt_dir):
        raise FileNotFoundError(f"Dir not found: {args.ckpt_dir}")

    files = [f for f in os.listdir(args.ckpt_dir) if f.endswith('.pth')]
    files.sort()

    results = []

    # 3. Iterate
    for f_name in files:
        f_path = os.path.join(args.ckpt_dir, f_name)

        try:
            params = parse_model_params(f_name)
        except Exception as e:
            print(f"Skipping {f_name}: Parse error {e}")
            continue

        print(f"\nProcessing: {f_name}")
        print(f"   -> Params: Z={params['z_dim']}, D={params['depth']}, IO={params['ioloss']}")

        try:
            model = build_model(args, params)
            state_dict = torch.load(f_path, map_location=device)

            if list(state_dict.keys())[0].startswith('module.') and not list(model.state_dict().keys())[0].startswith(
                    'module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)

            # Eval
            res = evaluate_single_model(model, test_dataloader, device, params, args, f_path)

            if res:
                results.append(res)
                print(
                    f"   -> Results: Acc: {res['Accuracy']:.4f} | AUC: {res['AUC_Mean']:.4f} | AUC_std: {res['AUC_Std']:.4f}")
                print(f"   -> LABROC file saved: {f_name.replace('.pth', '_labroc.txt')}")
            else:
                print("   -> Failed (likely missing NPY stats)")

        except Exception as e:
            print(f"   -> Error processing model: {e}")
            import traceback
            traceback.print_exc()

    # 4. Save Summary
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by='AUC_Mean', ascending=False)
        out_csv = os.path.join(args.ckpt_dir, 'test_summary_results.csv')
        df.to_csv(out_csv, index=False)
        print("\n" + "=" * 50)
        print(f"Saved results to {out_csv}")
        print(df[['Filename', 'Accuracy', 'AUC_Mean', 'AUC_Std', 'z_dim']].head().to_string())


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        gpu_id = sys.argv[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        sys.argv = [sys.argv[0]] + sys.argv[2:]

    args = get_args()
    main(args)