#!/usr/bin/env python
import os
import numpy as np
import h5py
from numpy.fft import fft2, ifft2

# ================== Basic Configuration ==================
# Image dimension（MRI data 260 x 311）
H, W = 260, 311  # 272 * 320

# Path Configuration
train_bg_path = "/shared/anastasio-s2/SOM/~"
test_bg_path  = "/shared/anastasio-s2/SOM/~"
loc_path      = "/shared/anastasio-s2/SOM/~"

# Original Parameters
mode           = 'sks'   # Signal Known Statistically (SKS)
min_sigma      = 3.0
max_sigma      = 3.0
noise_level    = 25.0
min_amplitude  = 0.2
max_amplitude  = 0.2

# For SKE/SKS: Using fixed sigma and amplitude for now
SIGMA = min_sigma        # Gaussian width
AMPL  = min_amplitude    # Gaussian amplitude

unbalanced = False       # We are doing balanced: equal number of positive and negative samples
# shuffle    = True        # Reflected in naming only; we control actual order manually

root_output_dir = "/shared/anastasio-s2/SI/HCP_selected"


# ================== Utility Function: Save to h5 ==================
def save_to_h5(path, data_dict):
    """Simple h5 save function: each key in the dict becomes a dataset."""
    with h5py.File(path, "w") as f:
        for k, v in data_dict.items():
            f.create_dataset(k, data=v)


# ================== Core Function: Generate split from background pairs ==================
def build_split_from_background_pair(background_data, loc, ampl, sigma, noise_level, mask=None, seed=None):
    """
    Input:
        background_data: (N, H, W) Pure backgrounds
        loc: (N_loc, 2) Signal coordinates
        ampl: Gaussian signal amplitude
        sigma: Gaussian signal width
    Output:
        data_with_signal: (2N, H, W), image domain:
            Even index 2*i   : No signal (Pure background)
            Odd index 2*i+1  : With signal (Background + Gaussian signal)
        invHg_data:       (2N, H, W), image domain (Reconstruction from full k-space + complex noise)
        label:            (2N,)：
            Index 2*i   -> 0 = No signal
            Index 2*i+1 -> 1 = With signal
        mask:             (H, W) Mask (All ones here)
    """
    if seed is not None:
        np.random.seed(seed)

    background_data = background_data.reshape(-1, H, W).astype(np.float32)
    N = background_data.shape[0]

    # Output size is 2N: Generate two samples (neg, pos) per background image
    data_with_signal = np.empty((2 * N, H, W), dtype=np.float32)
    invHg_data       = np.empty((2 * N, H, W), dtype=np.float32)
    label            = np.empty(2 * N, dtype=int)

    # Grid coordinates
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    # No undersampling -> mask is all ones
    if mask is None:
        mask = np.ones((H, W), dtype=np.float32)

    for i in range(N):
        bg = background_data[i].copy()

        # ========= 1) Negative Sample: No signal =========
        idx_neg = 2 * i
        img_neg = bg.copy()          # Do not add signal in image domain
        data_with_signal[idx_neg] = img_neg
        label[idx_neg] = 0           # No signal

        # Add complex noise in k-space
        kspace_neg = fft2(img_neg)
        noise_real_neg = np.random.normal(0, noise_level, size=kspace_neg.shape)
        noise_imag_neg = np.random.normal(0, noise_level, size=kspace_neg.shape)
        noise_neg = noise_real_neg + 1j * noise_imag_neg
        kspace_neg_noisy = kspace_neg + noise_neg
        kspace_neg_noisy = kspace_neg_noisy * mask

        img_neg_recon = ifft2(kspace_neg_noisy)
        invHg_data[idx_neg] = np.abs(img_neg_recon).astype(np.float32)

        # ========= 2) Positive Sample: With signal =========
        idx_pos = 2 * i + 1

        # Randomly select a location to add the Gaussian signal
        loc_index = np.random.randint(len(loc))
        y0 = loc[loc_index, 0]  # row
        x0 = loc[loc_index, 1]  # col

        signal = ampl * np.exp(
            -0.5 * (((X - x0) ** 2 + (Y - y0) ** 2) / (sigma ** 2))
        )
        img_pos = bg + signal.astype(np.float32)

        data_with_signal[idx_pos] = img_pos
        label[idx_pos] = 1        # With signal

        # Add complex noise in k-space
        kspace_pos = fft2(img_pos)
        noise_real_pos = np.random.normal(0, noise_level, size=kspace_pos.shape)
        noise_imag_pos = np.random.normal(0, noise_level, size=kspace_pos.shape)
        noise_pos = noise_real_pos + 1j * noise_imag_pos
        kspace_pos_noisy = kspace_pos + noise_pos
        kspace_pos_noisy = kspace_pos_noisy * mask

        img_pos_recon = ifft2(kspace_pos_noisy)
        invHg_data[idx_pos] = np.abs(img_pos_recon).astype(np.float32)

    # Samples are already in [neg0, pos0, neg1, pos1, ...] alternating order,
    # so no additional interleave operation is needed.
    return data_with_signal, invHg_data, label, mask

def global_minmax_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

# ================== Main Workflow ==================
def main():
    # ---- 1. Load Data ----
    print("Loading backgrounds...")
    bg_train = np.load(train_bg_path) # Shape usually (N_train, 260, 311)
    bg_test  = np.load(test_bg_path)

    loc = np.load(loc_path)  # (N_loc, 2)

    # ---- 2. Construct mask (no undersampling -> all ones) ----
    mask = np.ones((H, W), dtype=np.float32)

    # ---- 3. Generate Training Set ----
    print("Building TRAIN split (pair duplicates)...")
    data_with_signal_train, invHg_data_train, label_train, mask_train = build_split_from_background_pair(
        bg_train, loc, AMPL, SIGMA, noise_level, mask=mask, seed=123
    )

    # ---- 4. Generate Validation + Test (based on same background_test) ----
    bg_test = bg_test.reshape(-1, H, W).astype(np.float32)
    N_test_bg = bg_test.shape[0]
    mid = N_test_bg // 2
    bg_val  = bg_test[:mid]
    bg_test2 = bg_test[mid:]

    print("Building VAL split (pair duplicates)...")
    data_with_signal_val, invHg_data_val, label_val, mask_val = build_split_from_background_pair(
        bg_val, loc, AMPL, SIGMA, noise_level, mask=mask, seed=456
    )

    print("Building TEST split (pair duplicates)...")
    data_with_signal_test, invHg_data_test, label_test, mask_test = build_split_from_background_pair(
        bg_test2, loc, AMPL, SIGMA, noise_level, mask=mask, seed=789
    )


    # Normalize TRAIN
    data_with_signal_train = global_minmax_norm(data_with_signal_train)
    invHg_data_train       = global_minmax_norm(invHg_data_train)

    # Normalize VAL
    data_with_signal_val = global_minmax_norm(data_with_signal_val)
    invHg_data_val       = global_minmax_norm(invHg_data_val)

    # Normalize TEST
    data_with_signal_test = global_minmax_norm(data_with_signal_test)
    invHg_data_test       = global_minmax_norm(invHg_data_test)

    # ---- 5. File Naming (using original rules) ----
    if unbalanced:
        filename = '{}_{}_{}_{}_{}_c2_num_signals_diffusion_n'.format(mode, min_sigma, min_amplitude, noise_level, 'ub')
    else:
        filename = '{}_{}_{}_{}_c2_num_signals_diffusion_n'.format(mode, min_sigma, min_amplitude, noise_level)

    # if not shuffle:
    #     filename += '_noshuffle'

    print("filename: ", filename)
    output_dir = os.path.join(root_output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)

    # ---- 6. Assemble splits and save h5 ----
    splits = {
        'train': {
            'data_with_signal': data_with_signal_train,
            'invHg_data': invHg_data_train,
            'label': label_train,
            'mask': mask_train,
        },
        'val': {
            'data_with_signal': data_with_signal_val,
            'invHg_data': invHg_data_val,
            'label': label_val,
            'mask': mask_val,
        },
        'test': {
            'data_with_signal': data_with_signal_test,
            'invHg_data': invHg_data_test,
            'label': label_test,
            'mask': mask_test,
        }
    }

    for split_name, split_data in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        output_path = os.path.join(split_dir, 'data.h5')
        save_to_h5(output_path, split_data)
        print(f"Saved {split_name} to {output_path}")


if __name__ == "__main__":
    main()
