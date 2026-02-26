import h5py
import numpy as np
import os

# Run this once to convert your data
def convert_h5_to_npy(h5_path, output_dir):
    with h5py.File(h5_path, "r") as f:
        # Load data (chunk by chunk if too large, or all at once if possible)
        print("Loading HDF5...")
        data = f["data_with_signal"][:]
        labels = f["label"][:]
        measures = f["invHg_data"][:]

        print("Saving to .npy (this might take a moment)...")
        # Save as standard numpy files
        np.save(f"{output_dir}/data.npy", data)
        np.save(f"{output_dir}/label.npy", labels)
        np.save(f"{output_dir}/measure.npy", measures)
        print("Done!")


if __name__ == '__main__':
    # base_path_train = '/shared/anastasio-s2/SI/HCP_selected/sks_3.0_0.2_25.0_c2_num_signals_diffusion_n/train'
    base_path_valid = '/shared/anastasio-s2/SI/HCP_selected/sks_3.0_0.2_25.0_c2_num_signals_diffusion_n/val'
    base_path_test = '/shared/anastasio-s2/SI/HCP_selected/sks_3.0_0.2_25.0_c2_num_signals_diffusion_n/test'

    # h5_path_train = os.path.join(base_path_train, 'data.h5')
    h5_path_valid = os.path.join(base_path_valid, 'data.h5')
    h5_path_test = os.path.join(base_path_test, 'data.h5')

    # output_dir_train = base_path_train
    output_dir_valid = base_path_valid
    output_dir_test = base_path_test

    # convert_h5_to_npy(h5_path_train, output_dir_train)
    convert_h5_to_npy(h5_path_valid, output_dir_valid)
    convert_h5_to_npy(h5_path_test, output_dir_test)