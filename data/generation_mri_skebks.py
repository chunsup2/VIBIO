#!/usr/bin/env python
import os
import numpy as np
import h5py
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
# ================== Basic Configuration ==================
# Image dimensions (Your MRI data is 260 x 311)
H, W = 260, 311

# Path Configuration
train_bg_path = "/shared/anastasio-s2/SOM/kaiyan/mri_1000_200k/background_train.npy"
test_bg_path  = "/shared/anastasio-s2/SOM/kaiyan/mri_1000_200k/background_test.npy"
loc_path      = "/home/zexinji/Zexin/Code/VIB/vib-main/preprocessing/mri_loc.npy"

# Original Parameters
mode           = 'ske'   
min_sigma      = 3.0
max_sigma      = 3.0
noise_level    = 35.0
min_amplitude  = 0.05
max_amplitude  = 0.05

# For SKE/SKS: Use fixed sigma and amplitude for now
SIGMA = min_sigma        # Gaussian width
AMPL  = min_amplitude    # Gaussian amplitude

unbalanced = False       # We are doing "balanced": equal number of positive and negative samples
# shuffle    = True        # This is only reflected in the naming; we control the actual order manually

# Root Output Directory
root_output_dir = "/shared/anastasio-s2/SI/HCP_selected"



# ================== New: Save Example Images ==================
def save_example_images(data_with_signal, invHg_data, label, output_dir, num_examples=10):
    """
        Select num_examples samples with signals (label == 1) from the generated data,
        and save the following separately:
            - Background image (corresponding negative sample)
            - Clean image after adding signal (data_with_signal)
            - Reconstructed image after adding noise (invHg_data)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all positive sample indices
    pos_indices = np.where(label == 1)[0]
    if len(pos_indices) == 0:
        print("No positive samples (label=1) found, cannot save example images.")
        return

    pos_indices = pos_indices[:num_examples]

    for i, pos_idx in enumerate(pos_indices):
        # Positive index = 2 * bg_idx + 1
        # Negative index = 2 * bg_idx
        bg_idx = pos_idx // 2
        neg_idx = 2 * bg_idx

        img_bg    = data_with_signal[neg_idx]   # Pure background
        img_clean = data_with_signal[pos_idx]   # Background + Signal (No noise yet)
        img_noisy = invHg_data[pos_idx]         # Reconstructed image after adding noise
        signal = img_clean - img_bg

        # Save as png
        plt.imsave(os.path.join(output_dir, f"example_{i:02d}_bg.png"),
                   img_bg, cmap="gray")
        plt.imsave(os.path.join(output_dir, f"example_{i:02d}_clean.png"),
                   img_clean, cmap="gray")
        plt.imsave(os.path.join(output_dir, f"example_{i:02d}_noisy.png"),
                   img_noisy, cmap="gray")
        plt.imsave(os.path.join(output_dir, f"example_{i:02d}_signal.png"),
                   signal, cmap="gray")


    print(f"Saved {len(pos_indices)} sets of example images (3 images per set) in {output_dir}.")


# ================== Utility Function: Save to h5 ==================
def save_to_h5(path, data_dict):
    """Simple h5 save function: each key in the dict becomes a dataset."""
    with h5py.File(path, "w") as f:
        for k, v in data_dict.items():
            f.create_dataset(k, data=v)


# ================== Core Function: Build a split from background pairs ==================
def build_split_from_background_pair(background_data, loc, ampl, sigma, noise_level, mask=None, seed=None):
    """
    Input:
        background_data: (N, H, W) Pure backgrounds
        loc: (N_loc, 2) Signal coordinates
        ampl: Gaussian signal amplitude
        sigma: Gaussian signal width
    Output:
        data_with_signal: (2N, H, W), Image domain:
            Even index 2*i  : No signal (Pure background)
            Odd index 2*i+1 : With signal (Background + Gaussian signal)
        invHg_data:       (2N, H, W), Image domain (Reconstructed image after k-space complex noise)
        label:            (2N,):
            2*i   position -> 0 = No signal
            2*i+1 position -> 1 = With signal
        mask:             (H, W) Mask (Full ones here)
    """
    if seed is not None:
        np.random.seed(seed)

    background_data = background_data.reshape(-1, H, W).astype(np.float32)
    N = background_data.shape[0]

    # Output size is 2N: generate two samples (neg, pos) for each background
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

        # ========= 1) Negative Sample: No Signal =========
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

        # ========= 2) Positive Sample: With Signal =========
        idx_pos = 2 * i + 1

        # Select a location to add Gaussian signal (currently hardcoded to 170, 180)
        # loc_index = np.random.randint(len(loc))
        # y0 = loc[loc_index, 0]  # row
        # x0 = loc[loc_index, 1]  # col
        y0 = 170
        x0 = 180

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

    # It is already in [neg0, pos0, neg1, pos1, ...] alternating order,
    # so no additional interleave operation is required.
    return data_with_signal, invHg_data, label, mask

# ===== Global min-max normalization for each split =====
def global_minmax_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

# ================== Main Process ==================
def main():
    # ---- 1. Load Data ----
    print("Loading backgrounds...")
    # bg_train = np.load(train_bg_path)  # Shape is usually (N_train, 260, 311)
    bg_test  = np.load(test_bg_path)

    loc = np.load(loc_path)  # (N_loc, 2)

    # ---- 2. Construct mask (No undersampling -> all ones) ----
    mask = np.ones((H, W), dtype=np.float32)

    # ---- 3. Generate training set ----
    # print("Building TRAIN split (pair duplicates)...")
    # data_with_signal_train, invHg_data_train, label_train, mask_train = build_split_from_background_pair(
    #     bg_train, loc, AMPL, SIGMA, noise_level, mask=mask, seed=123
    # )

    # ---- 4. Generate Val + Test, both based on the same background_test ----
    bg_test = bg_test.reshape(-1, H, W).astype(np.float32)
    N_test_bg = bg_test.shape[0]
    mid = N_test_bg // 2
    bg_val  = bg_test[:mid]
    bg_test2 = bg_test[mid:]

    print("Building VAL split (pair duplicates)...")
    data_with_signal_val, invHg_data_val, label_val, mask_val = build_split_from_background_pair(
        bg_val, loc, AMPL, SIGMA, noise_level, mask=mask, seed=456
    )
    # bg_test = bg_test.reshape(-1, H, W).astype(np.float32)

    print("Building VAL+TEST split (same backgrounds)...")
    # data_with_signal_val, invHg_data_val, label_val, mask_val = build_split_from_background_pair(
    #     bg_test, loc, AMPL, SIGMA, noise_level, mask=mask, seed=456
    # )

    # print("Building TEST split (pair duplicates)...")
    data_with_signal_test, invHg_data_test, label_test, mask_test = build_split_from_background_pair(
        bg_test2, loc, AMPL, SIGMA, noise_level, mask=mask, seed=789
    )

    # If you want val/test to be identical, directly copy them
    # data_with_signal_test = data_with_signal_val.copy()
    # invHg_data_test       = invHg_data_val.copy()
    # label_test            = label_val.copy()
    # mask_test             = mask_val  # mask is all ones, copying doesn't matter much

    # Normalize TRAIN
    # data_with_signal_train = global_minmax_norm(data_with_signal_train)
    # invHg_data_train       = global_minmax_norm(invHg_data_train)

    # Normalize VAL
    data_with_signal_val = global_minmax_norm(data_with_signal_val)
    invHg_data_val       = global_minmax_norm(invHg_data_val)

    # Normalize TEST
    data_with_signal_test = global_minmax_norm(data_with_signal_test)
    invHg_data_test       = global_minmax_norm(invHg_data_test)

    # ---- 5. File naming (Following your original rules) ----
    if unbalanced:
        filename = '{}_{}_{}_{}_{}_c2_num_signals_diffusion_n_new'.format(mode, min_sigma, min_amplitude, noise_level, 'ub')
    else:
        filename = '{}_{}_{}_{}_c2_num_signals_diffusion_n_new'.format(mode, min_sigma, min_amplitude, noise_level)

    # if not shuffle:
    #     filename += '_noshuffle'

    print("filename: ", filename)
    output_dir = os.path.join(root_output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)

    # ---- 6. Assemble splits and save h5 ----
    splits = {
        # 'train': {
        #     'data_with_signal': data_with_signal_train,
        #     'invHg_data': invHg_data_train,
        #     'label': label_train,
        #     'mask': mask_train,
        # },
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

    
     # ---- 7. Extra: Save 10 sets of example images from train ----
    # finaloutput_dir = '/home/zexinji/Zexin/Code/VIB/vib-main/preprocessing'
    # example_dir = os.path.join(finaloutput_dir, "train_examples")
    # save_example_images(
    #     data_with_signal_train,
    #     invHg_data_train,
    #     label_train,
    #     example_dir,
    #     num_examples=10
    # )


if __name__ == "__main__":
    main()
