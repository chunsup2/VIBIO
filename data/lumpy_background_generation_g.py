import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import os

def generate_lumpy_background(image_size=64, N_mean=5, a=1, s=7, w=0.5, h=40):
    Nb = np.random.poisson(N_mean)
    lump_centers = np.random.uniform(0, image_size, size=(Nb, 2))
    x = np.arange(image_size)
    y = np.arange(image_size)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx, yy], axis=-1)
    denom = 2 * (w**2 + s**2)
    coeff = a * h * s**2 / (w**2 + s**2)
    b = np.zeros((image_size, image_size))
    for rn in lump_centers:
        dist2 = np.sum((grid - rn) ** 2, axis=-1)
        b += np.exp(-dist2 / denom)
    return coeff * b

def generate_signal_image(image_size=64, A=0.2, rc=(32, 32), ws=3, w=0.5, h=40):
    x = np.arange(image_size)
    y = np.arange(image_size)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx, yy], axis=-1)
    dist2 = np.sum((grid - rc) ** 2, axis=-1)
    denom = 2 * (w**2 + ws**2)
    coeff = A * h * ws**2 / (w**2 + ws**2)
    return coeff * np.exp(-dist2 / denom)

def add_gaussian_noise(image, std=20):
    """
    Adds i.i.d. Gaussian noise N(0, δ^2) to the image.
    δ = std = 20 by default, following the paper.
    """
    noise = np.random.normal(loc=0.0, scale=std, size=image.shape)
    return image + noise

def generate_dataset_split(num_images, image_size, A, rc, ws, std, signal_present, params):
    data = []
    labels = []
    for _ in tqdm(range(num_images)):
        b = generate_lumpy_background(image_size, **params)
        if signal_present:
            s = generate_signal_image(image_size, A, rc, ws, params['w'], params['h'])
            img = b + s
            label = 1
        else:
            img = b
            label = 0
        data.append(add_gaussian_noise(img, std))
        labels.append(label)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.uint8)

def interleave_signal_absent_present(data_pos, labels_pos, data_neg, labels_neg):
    interleaved_data = np.empty((data_pos.shape[0] + data_neg.shape[0], *data_pos.shape[1:]), dtype=data_pos.dtype)
    interleaved_labels = np.empty(data_pos.shape[0] + data_neg.shape[0], dtype=labels_pos.dtype)
    interleaved_data[0::2], interleaved_data[1::2] = data_pos, data_neg
    interleaved_labels[0::2], interleaved_labels[1::2] = labels_pos, labels_neg
    return interleaved_data, interleaved_labels

def save_to_h5(output_path, data_dict):
    with h5py.File(output_path, 'w') as hf:
        for key, data in data_dict.items():
            hf.create_dataset(key, data=data)
    print(f"Data saved to {output_path}")

if __name__ == '__main__':
    np.random.seed(0)

    # Parameters
    image_size = 64
    A = 0.2
    rc = (32, 32)
    ws = 3
    std = 30
    #std = 20

    # Dataset sizes: Modify to 200,000 / 20,000 / 20,000 as needed
    # num_train = 1000000
    # num_train = 10
    num_train = 1000000
    num_test  = 20000
    num_val   = 20000

    background_params = {
        'N_mean': 5,
        'a': 1,
        's': 7,
        'w': 0.5,
        'h': 40
    }

    # Generate two versions: normalize=True and normalize=False
    for normalize in [True,False]:
        if not normalize:
            filename = f"lumpy_background_A{A}_std{std}_num{num_train}_num{num_test}_original_gau"
        else:
            filename = f"lumpy_background_A{A}_std{std}_num{num_train}_num{num_test}_gau"

        # output_dir = f'./lumpy/{filename}'
        output_dir = f'/shared/anastasio-s2/SI/HCP_selected/{filename}'
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n=== Generating dataset version: normalize={normalize} ===")
        for split_name, num_images in zip(['train', 'test', 'val'], [num_train, num_test, num_val]):
            print(f"Generating {split_name} data...")

            data_pos, labels_pos = generate_dataset_split(
                num_images, image_size, A, rc, ws, std, True, background_params
            )
            data_neg, labels_neg = generate_dataset_split(
                num_images, image_size, A, rc, ws, std, False, background_params
            )

            data, labels = interleave_signal_absent_present(data_pos, labels_pos, data_neg, labels_neg)

            # Normalize using global mean/std of the current split
            if normalize:
                mu = data.mean(dtype=np.float64)
                sigma = data.std(dtype=np.float64) + 1e-8
                data = ((data - mu) / sigma).astype(np.float32)
                print(f"{split_name}: mean={mu:.4f}, std={sigma:.4f}")

            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            save_to_h5(os.path.join(split_dir, 'data.h5'), {
                'data_with_signal': data,
                'labels': labels
            })

        print(f"Version (normalize={normalize}) done! Output -> {output_dir}")

    print("\nAll datasets generated successfully.")
