import numpy as np
import matplotlib.pyplot as plt  # 可用于快速可视化
from tqdm import tqdm
import h5py
import os
import json

# -----------------------------
# 1) Background: SKE/BKE → b = 0 (Keep function name/interface, always return zeros)
# -----------------------------
def generate_lumpy_background(image_size=64, N_mean=5, a=1, s=7, w=0.5, h=40):
    """
    SKE/BKE Setting: Background is non-random and known, b = 0.
    Maintained for compatibility with original code, but consistently returns zero background.
    """
    return np.zeros((image_size, image_size), dtype=np.float32)

# -----------------------------
# 2) Signal: 2D Symmetric Gaussian (Equation (27) form, including w, h)
#    s_m = A*h*ws^2/(w^2+ws^2) * exp(-||r_m-rc||^2 / (2*(w^2+ws^2)))
# -----------------------------
def generate_signal_image(image_size=64, A=0.2, rc=(32, 32), ws=3, w=0.5, h=40):
    x = np.arange(image_size, dtype=np.float32)
    y = np.arange(image_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    # rc is given in pixel coordinates by default
    dist2 = (xx - rc[0])**2 + (yy - rc[1])**2
    denom = 2.0 * (w**2 + ws**2)
    coeff = A * h * (ws**2) / (w**2 + ws**2)
    s = coeff * np.exp(-dist2 / denom)
    return s.astype(np.float32)


# -----------------------------
# 3) Gaussian Noise: δ = std
# -----------------------------
def add_gaussian_noise(image, std=20.0):
    noise = np.random.normal(loc=0.0, scale=std, size=image.shape).astype(np.float32)
    return (image.astype(np.float32) + noise).astype(np.float32)

# -----------------------------
# 4) Generate single class data (H1/H0) and return IO-LR (Gaussian case)
#    Noise: n ~ N(0, σ^2 I)
#    log LR(g) = (1/σ^2) * ( s^T g - 0.5 * s^T s )   (b=0)
#    LR(g) = exp(log LR(g))
# -----------------------------
def generate_dataset_split(num_images, image_size, A, rc, ws, std, signal_present, params):
    data = []
    labels = []
    io_lr = []
    log_io_lr = []

    # Pre-generate deterministic signal template (reusable)
    s = generate_signal_image(image_size, A, rc, ws, params['w'], params['h']).astype(np.float32)
    s64 = s.astype(np.float64)
    s_dot_s = float(np.sum(s64 * s64))  # s^T s

    # b=0（SKE/BKE）
    _ = generate_lumpy_background(image_size, **params)  # Interface consistency; unused

    inv_sigma2 = 1.0 / (std * std)

    for _ in tqdm(range(num_images)):
        if signal_present:
            img_clean = s               # H1: g = s + n
            label = 1
        else:
            img_clean = np.zeros_like(s)  # ✅ FIX: H0 must be an all-zero image of the same size as s
            label = 0

        # Observation g
        g = add_gaussian_noise(img_clean, std).astype(np.float32)

        # Calculate log LR and LR (Gaussian closed-form solution)
        g64 = g.astype(np.float64)
        s_dot_g = float(np.sum(s64 * g64))
        loglr = inv_sigma2 * (s_dot_g - 0.5 * s_dot_s)

        # Prevent overflow when generating LR; simultaneously save logLR (recommended for evaluation)
        if loglr <= -700:
            lr = 0.0
        elif loglr >= 700:
            lr = np.inf
        else:
            lr = float(np.exp(loglr))

        data.append(g)
        labels.append(label)
        io_lr.append(lr)
        log_io_lr.append(loglr)

    return (np.array(data, dtype=np.float32),
            np.array(labels, dtype=np.uint8),
            np.array(io_lr, dtype=np.float64),
            np.array(log_io_lr, dtype=np.float64))

# -----------------------------
# 5) Interleave positive and negative samples (Maintains your logic, adds logLR)
# -----------------------------
def interleave_signal_absent_present(data_pos, labels_pos, io_pos, logio_pos,
                                     data_neg, labels_neg, io_neg, logio_neg):
    N = data_pos.shape[0] + data_neg.shape[0]
    interleaved_data   = np.empty((N, *data_pos.shape[1:]), dtype=data_pos.dtype)
    interleaved_labels = np.empty((N,), dtype=labels_pos.dtype)
    interleaved_iolr   = np.empty((N,), dtype=io_pos.dtype)
    interleaved_loglr  = np.empty((N,), dtype=logio_pos.dtype)

    interleaved_data[0::2],   interleaved_data[1::2]   = data_pos,   data_neg
    interleaved_labels[0::2], interleaved_labels[1::2] = labels_pos, labels_neg
    interleaved_iolr[0::2],   interleaved_iolr[1::2]   = io_pos,     io_neg
    interleaved_loglr[0::2],  interleaved_loglr[1::2]  = logio_pos,  logio_neg
    return interleaved_data, interleaved_labels, interleaved_iolr, interleaved_loglr

# -----------------------------
# 6) Save HDF5: gzip compression + metadata
# -----------------------------
def save_to_h5(output_path, data_dict, attrs=None):
    with h5py.File(output_path, 'w') as hf:
        for key, data in data_dict.items():
            hf.create_dataset(
                key, data=data
            )
        if attrs is not None:
            for k, v in attrs.items():
                hf.attrs[k] = json.dumps(v)
    print(f"Data saved to {output_path}")

# -----------------------------
# 7) Main Script
# -----------------------------


if __name__ == '__main__':
    np.random.seed(0)

    image_size = 64
    A = 0.2
    rc = (32, 32)
    ws = 3
    # std = 20
    std = 40
    # num_train = 10
    # num_train = 250000
    num_train = 500000
    num_test  = 20000
    num_val   = 20000

    background_params = {
        'N_mean': 5,
        'a': 1,
        's': 7,
        'w': 0.5,
        'h': 40
    }

    # ----------------------------
    # Dual version output: normalize=True and False
    # ----------------------------
    for normalize in [True,False]:
        tag = "" if normalize else "_original"
        filename = f"ske_bke_gauss_A{A}_std{std}_num{num_train}_num{num_test}{tag}"
        output_dir = f'/shared/anastasio-s2/SI/HCP_selected/{filename}'
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n=== Generating dataset version: normalize={normalize} ===")

        for split_name, num_images in zip(['train', 'test', 'val'],
                                          [num_train, num_test, num_val]):
            print(f"\nGenerating {split_name} data...")

            # Generate H1 and H0
            data_pos, labels_pos, io_pos, logio_pos = generate_dataset_split(
                num_images, image_size, A, rc, ws, std, True, background_params
            )
            data_neg, labels_neg, io_neg, logio_neg = generate_dataset_split(
                num_images, image_size, A, rc, ws, std, False, background_params
            )

            # Combine and Interleave
            data, labels, io_lr, log_io_lr = interleave_signal_absent_present(
                data_pos, labels_pos, io_pos, logio_pos,
                data_neg, labels_neg, io_neg, logio_neg
            )

            # Normalization check
            if normalize:
                mu = data.mean(dtype=np.float64)
                sigma = data.std(dtype=np.float64) + 1e-8
                data = ((data - mu) / sigma).astype(np.float32)
                print(f"{split_name}: mean={mu:.4f}, std={sigma:.4f}")

            # Output directory
            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            attrs = {
                "task": "SKE/BKE",
                "image_size": image_size,
                "background": "zeros (b=0)",
                "signal": {
                    "type": "2D Gaussian (with w,h)",
                    "A": A, "rc": rc, "ws": ws,
                    "formula": "s_m = A*h*ws^2/(w^2+ws^2) * exp(-||r_m-rc||^2/(2*(w^2+ws^2)))",
                    "params_w_h_from_background_params": True
                },
                "noise": {"distribution": "Gaussian", "mean": 0.0, "std": std},
                "counts": {"per_class": int(num_images), "total": int(2*num_images)},
                "normalize": normalize,
                "equations": {
                    "IO_logLR_Gaussian": "log LR(g) = (1/sigma^2) * (s^T g - 0.5 * s^T s), with b=0",
                    "IO_LR_Gaussian": "LR(g) = exp(log LR(g))"
                }
            }

            save_to_h5(os.path.join(split_dir, 'data.h5'), {
                'data_with_signal': data,
                'labels': labels,
                'io_lr': io_lr,
                'log_io_lr': log_io_lr,
            }, attrs=attrs)

        print(f"Version (normalize={normalize}) done! Output -> {output_dir}")

    print("\nAll datasets generated successfully.")
