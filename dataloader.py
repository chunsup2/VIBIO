import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import os
import time
import psutil

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

import torch.nn.functional as F
from torch.utils.data import Dataset, get_worker_info


# class MRIDataset1(Dataset):
#     def __init__(
#         self,
#         h5_file_path,
#         proportion: float = 1.0,
#         use_padding: bool = True,
#         target_size=(272, 320),
#         pad_mode: str = "constant",
#         pad_value: float = 0.0,
#     ):
#         super().__init__()

#         self.h5_file_path = h5_file_path
#         self.h5_file = None          # 先不打开
#         self.data_with_signal_ds = None
#         self.invHg_data_ds = None
#         self.label_ds = None

#         self.proportion = proportion
#         self.use_padding = use_padding
#         self.target_size = target_size if use_padding else None
#         self.pad_mode = pad_mode
#         self.pad_value = pad_value

#         # 先默认这些值，真正的值会在第一次 _lazy_init 时重新计算
#         self.total_length = None
#         self.selected_length = None
#         self.H = None
#         self.W = None

#         # 先把 padding 设成 0，_lazy_init 里会重算
#         self.pad_top = self.pad_bottom = self.pad_left = self.pad_right = 0

#     def _lazy_init(self):
#         """在第一次访问时打开 h5 文件（每个 worker 进程各自执行一次）"""
#         if self.h5_file is not None:
#             return

#         self.h5_file = h5py.File(self.h5_file_path, "r")

#         self.data_with_signal_ds = self.h5_file["data_with_signal"]
#         self.invHg_data_ds = self.h5_file["invHg_data"]
#         self.label_ds = self.h5_file["label"]

#         self.total_length = int(len(self.data_with_signal_ds))
#         self.selected_length = int(self.total_length * self.proportion)

#         _, self.H, self.W = self.data_with_signal_ds.shape
#         print(f"[MRIDataset] opened h5, original size: ({self.H}, {self.W}), total={self.total_length}")

#         if self.target_size is not None:
#             target_H, target_W = self.target_size
#             assert target_H >= self.H and target_W >= self.W, \
#                 f"target_size {self.target_size} 必须 >= 原始尺寸 ({self.H}, {self.W})"

#             pad_h_total = target_H - self.H
#             pad_w_total = target_W - self.W

#             self.pad_top = pad_h_total // 2
#             self.pad_bottom = pad_h_total - self.pad_top
#             self.pad_left = pad_w_total // 2
#             self.pad_right = pad_w_total - self.pad_left

#             print(
#                 f"[MRIDataset] will pad from ({self.H}, {self.W}) "
#                 f"to ({target_H}, {target_W}) with "
#                 f"(top={self.pad_top}, bottom={self.pad_bottom}, "
#                 f"left={self.pad_left}, right={self.pad_right}), "
#                 f"mode={self.pad_mode}"
#             )
#         else:
#             print(f"[MRIDataset] no padding, using original size ({self.H}, {self.W})")

#     def __len__(self):
#         # 如果 len 在 _lazy_init 之前被调用，可以给个兜底
#         if self.selected_length is None:
#             # 粗暴方案：临时打开一下，只为拿长度，然后关掉
#             with h5py.File(self.h5_file_path, "r") as f:
#                 total_length = int(len(f["data_with_signal"]))
#             return int(total_length * self.proportion)
#         return self.selected_length

#     def __getitem__(self, idx):
#         self._lazy_init()

#         image = self.data_with_signal_ds[idx]   # (H, W) numpy
#         measure = self.invHg_data_ds[idx]       # (H, W) numpy
#         task_label = self.label_ds[idx]         # 标量

#         # ========= 2. 把 torch.tensor(...) 换成 torch.from_numpy(...) =========
#         # 前提：HDF5 里保存的数据最好本来就是 float32 / int64

#         # 如果 HDF5 里 data 已经是 float32，可以直接 from_numpy，零拷贝
#         image = torch.from_numpy(image).float().unsqueeze(0)

#         # 如果 invHg_data 在存的时候已经 abs 过，就不要再 np.abs 了
#         # measure = np.abs(measure)   # 如果真的需要再 abs 再开
#         measure = torch.from_numpy(measure).abs().float().unsqueeze(0)

#         task_label = torch.as_tensor(task_label, dtype=torch.long)

#         if self.target_size is not None:
#             pad = (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
#             image = F.pad(image, pad, mode=self.pad_mode, value=self.pad_value)
#             measure = F.pad(measure, pad, mode=self.pad_mode, value=self.pad_value)

#         return image, measure, task_label

#     def close(self):
#         if getattr(self, "h5_file", None) is not None:
#             self.h5_file.close()
#             self.h5_file = None
#             print("HDF5 file closed")

#     def __del__(self):
#         try:
#             self.close()
#         except Exception:
#             pass

class MRIDataset1(Dataset):
    def __init__(
        self,
        h5_file_path,
        proportion: float = 1.0,
        use_padding: bool = True,
        target_size=(272, 320),
        pad_mode: str = "constant",
        pad_value: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.h5_file = None
        self.data_with_signal_ds = None
        self.invHg_data_ds = None
        self.label_ds = None

        self.proportion = proportion
        self.target_size = target_size if use_padding else None
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.verbose = verbose

        # ✅ 1) Read length in init to avoid repeated opening in __len__
        with h5py.File(self.h5_file_path, "r") as f:
            ds = f["data_with_signal"]
            self.total_length = int(len(ds))
            _, self.H, self.W = ds.shape

        self.selected_length = int(self.total_length * self.proportion)

        # ✅ 2) Pre-calculate pad parameters (meta-info read once)
        self.pad_top = self.pad_bottom = self.pad_left = self.pad_right = 0
        if self.target_size is not None:
            target_H, target_W = self.target_size
            assert target_H >= self.H and target_W >= self.W

            pad_h_total = target_H - self.H
            pad_w_total = target_W - self.W
            self.pad_top = pad_h_total // 2
            self.pad_bottom = pad_h_total - self.pad_top
            self.pad_left = pad_w_total // 2
            self.pad_right = pad_w_total - self.pad_left

        self._did_print = False  # Track printing status

    def _lazy_init(self):
        if self.h5_file is not None:
            return

        # ✅ Each worker opens its own handle (Correct for multi-processing)
        self.h5_file = h5py.File(self.h5_file_path, "r")
        self.data_with_signal_ds = self.h5_file["data_with_signal"]
        self.invHg_data_ds = self.h5_file["invHg_data"]
        self.label_ds = self.h5_file["label"]

        if self.verbose and not self._did_print:
            wi = get_worker_info()
            # Only allow worker 0 to print (avoiding cluttered output)
            if wi is None or wi.id == 0:
                print(f"[MRIDataset] opened h5: {self.h5_file_path}")
                print(f"[MRIDataset] original size: ({self.H}, {self.W}), total={self.total_length}, selected={self.selected_length}")
                if self.target_size is not None:
                    print(f"[MRIDataset] pad -> {self.target_size}, "
                          f"(top={self.pad_top}, bottom={self.pad_bottom}, left={self.pad_left}, right={self.pad_right}), mode={self.pad_mode}")
            self._did_print = True

    def __len__(self):
        return self.selected_length

    def __getitem__(self, idx):
        self._lazy_init()

        # (Optional) Ensure idx is within bounds
        if idx < 0 or idx >= self.selected_length:
            raise IndexError

        image = self.data_with_signal_ds[idx]    # numpy array
        measure = self.invHg_data_ds[idx]
        task_label = self.label_ds[idx]

        # Note: Copying occurs if the source data is not float32 (normal behavior)
        image = torch.from_numpy(image).to(torch.float32).unsqueeze(0)
        measure = torch.from_numpy(measure).to(torch.float32).abs().unsqueeze(0)
        task_label = torch.as_tensor(task_label, dtype=torch.long)

        if self.target_size is not None:
            pad = (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
            image = F.pad(image, pad, mode=self.pad_mode, value=self.pad_value)
            measure = F.pad(measure, pad, mode=self.pad_mode, value=self.pad_value)

        return image, measure, task_label

    def close(self):
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            finally:
                self.h5_file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class MRIDataset1_2(Dataset):
    def __init__(
        self,
        h5_file_path,
        proportion: float = 1.0,
        use_padding: bool = True,
        target_size=(272, 320),
        pad_mode: str = "constant",
        pad_value: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.h5_file = None
        self.data_with_signal_ds = None
        self.invHg_data_ds = None
        self.label_ds = None

        self.proportion = proportion
        self.target_size = target_size if use_padding else None
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.verbose = verbose

        # ✅ 1) Read length in init to avoid repeated opening in __len__
        with h5py.File(self.h5_file_path, "r") as f:
            ds = f["data_with_signal"]
            self.total_length = int(len(ds))
            _, self.H, self.W = ds.shape

        self.selected_length = int(self.total_length * self.proportion)

        # ✅ 2) Pre-calculate pad parameters (meta-info read once)
        self.pad_top = self.pad_bottom = self.pad_left = self.pad_right = 0
        if self.target_size is not None:
            target_H, target_W = self.target_size
            assert target_H >= self.H and target_W >= self.W

            pad_h_total = target_H - self.H
            pad_w_total = target_W - self.W
            self.pad_top = pad_h_total // 2
            self.pad_bottom = pad_h_total - self.pad_top
            self.pad_left = pad_w_total // 2
            self.pad_right = pad_w_total - self.pad_left

        self._did_print = False  # Track printing status

    def _lazy_init(self):
        if self.h5_file is not None:
            return

        # ✅ Each worker opens its own handle (Correct for multi-processing)
        self.h5_file = h5py.File(self.h5_file_path, "r") # 10 MB cache per worker
        # self.data_with_signal_ds = self.h5_file["data_with_signal"]
        self.invHg_data_ds = self.h5_file["invHg_data"]
        self.label_ds = self.h5_file["label"]

        if self.verbose and not self._did_print:
            wi = get_worker_info()
            # Only allow worker 0 to print (avoiding cluttered output)
            if wi is None or wi.id == 0:
                print(f"[MRIDataset] opened h5: {self.h5_file_path}")
                print(f"[MRIDataset] original size: ({self.H}, {self.W}), total={self.total_length}, selected={self.selected_length}")
                if self.target_size is not None:
                    print(f"[MRIDataset] pad -> {self.target_size}, "
                          f"(top={self.pad_top}, bottom={self.pad_bottom}, left={self.pad_left}, right={self.pad_right}), mode={self.pad_mode}")
            self._did_print = True

    def __len__(self):
        return self.selected_length

    def __getitem__(self, idx):
        self._lazy_init()

        # (Optional) Ensure idx is within bounds
        if idx < 0 or idx >= self.selected_length:
            raise IndexError

        # image = self.data_with_signal_ds[idx]    # numpy array
        measure = self.invHg_data_ds[idx]
        task_label = self.label_ds[idx]

        # Note: Copying occurs if the source data is not float32 (normal behavior)
        # image = torch.from_numpy(image).to(torch.float32).unsqueeze(0)
        measure = torch.from_numpy(measure).to(torch.float32).abs().unsqueeze(0)
        task_label = torch.as_tensor(task_label, dtype=torch.long)

        if self.target_size is not None:
            pad = (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
            # image = F.pad(image, pad, mode=self.pad_mode, value=self.pad_value)
            measure = F.pad(measure, pad, mode=self.pad_mode, value=self.pad_value)

        # return image, measure, task_label
        return measure, task_label

    def close(self):
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            finally:
                self.h5_file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class MRIDataset2(Dataset):
    def __init__(
            self,
            h5_file_path,
            proportion: float = 1.0,
            use_padding: bool = True,
            target_size=(272, 320),
            pad_mode: str = "constant",
            pad_value: float = 0.0,
            verbose: bool = False,
    ):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.proportion = proportion
        self.target_size = target_size if use_padding else None
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.verbose = verbose

        print(f"[MRIDataset] Loading data into RAM from {self.h5_file_path}...")

        # ✅ 1) Load EVERYTHING into RAM immediately
        with h5py.File(self.h5_file_path, "r") as f:
            # Get total length
            ds = f["data_with_signal"]
            total_length = int(len(ds))
            _, self.H, self.W = ds.shape

            # Calculate how many to load
            self.selected_length = int(total_length * self.proportion)

            if self.verbose:
                print(f"[MRIDataset] Reading {self.selected_length} samples (Proportion: {self.proportion})...")

            # Load the actual data into memory using slicing [:]
            # This creates numpy arrays in RAM
            self.data_with_signal = f["data_with_signal"][:self.selected_length]
            self.invHg_data = f["invHg_data"][:self.selected_length]
            self.label = f["label"][:self.selected_length]

        print("[MRIDataset] Finished loading data to RAM.")

        # ✅ 2) Pre-calculate pad parameters
        self.pad_top = self.pad_bottom = self.pad_left = self.pad_right = 0
        if self.target_size is not None:
            target_H, target_W = self.target_size
            assert target_H >= self.H and target_W >= self.W

            pad_h_total = target_H - self.H
            pad_w_total = target_W - self.W
            self.pad_top = pad_h_total // 2
            self.pad_bottom = pad_h_total - self.pad_top
            self.pad_left = pad_w_total // 2
            self.pad_right = pad_w_total - self.pad_left

    def __len__(self):
        return self.selected_length

    def __getitem__(self, idx):
        # ✅ No lazy_init needed, data is already in self.data_with_signal

        # (Optional) Ensure idx is within bounds
        if idx < 0 or idx >= self.selected_length:
            raise IndexError

        # ✅ Direct RAM access (very fast)
        image = self.data_with_signal[idx]
        measure = self.invHg_data[idx]
        task_label = self.label[idx]

        # Convert to Tensor
        image = torch.from_numpy(image).to(torch.float32).unsqueeze(0)
        measure = torch.from_numpy(measure).to(torch.float32).abs().unsqueeze(0)
        task_label = torch.as_tensor(task_label, dtype=torch.long)

        # Padding
        if self.target_size is not None:
            pad = (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
            image = F.pad(image, pad, mode=self.pad_mode, value=self.pad_value)
            measure = F.pad(measure, pad, mode=self.pad_mode, value=self.pad_value)

        return image, measure, task_label


class MRIDataset2_2(Dataset):
    def __init__(
            self,
            h5_file_path,
            proportion: float = 1.0,
            use_padding: bool = True,
            target_size=(272, 320),
            pad_mode: str = "constant",
            pad_value: float = 0.0,
            verbose: bool = False,
    ):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.proportion = proportion
        self.target_size = target_size if use_padding else None
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.verbose = verbose

        print(f"[MRIDataset] Loading data into RAM from {self.h5_file_path}...")

        # ✅ 1) Load EVERYTHING into RAM immediately
        with h5py.File(self.h5_file_path, "r") as f:
            # Get total length
            ds = f["label"]
            total_length = int(len(ds))

            # Calculate how many to load
            self.selected_length = int(total_length * self.proportion)

            if self.verbose:
                print(f"[MRIDataset] Reading {self.selected_length} samples (Proportion: {self.proportion})...")

            # Load the actual data into memory using slicing [:]
            # This creates numpy arrays in RAM
            # self.data_with_signal = f["data_with_signal"][:self.selected_length]
            self.invHg_data = f["invHg_data"][:self.selected_length]
            self.label = f["label"][:self.selected_length]

            _, self.H, self.W = self.invHg_data.shape

        print("[MRIDataset] Finished loading data to RAM.")

        # ✅ 2) Pre-calculate pad parameters
        self.pad_top = self.pad_bottom = self.pad_left = self.pad_right = 0
        if self.target_size is not None:
            target_H, target_W = self.target_size
            assert target_H >= self.H and target_W >= self.W

            pad_h_total = target_H - self.H
            pad_w_total = target_W - self.W
            self.pad_top = pad_h_total // 2
            self.pad_bottom = pad_h_total - self.pad_top
            self.pad_left = pad_w_total // 2
            self.pad_right = pad_w_total - self.pad_left

    def __len__(self):
        return self.selected_length

    def __getitem__(self, idx):
        # ✅ No lazy_init needed, data is already in self.data_with_signal

        # (Optional) Ensure idx is within bounds
        if idx < 0 or idx >= self.selected_length:
            raise IndexError

        # ✅ Direct RAM access (very fast)
        # image = self.data_with_signal[idx]
        measure = self.invHg_data[idx]
        task_label = self.label[idx]

        # Convert to Tensor
        # image = torch.from_numpy(image).to(torch.float32).unsqueeze(0)
        measure = torch.from_numpy(measure).to(torch.float32).abs().unsqueeze(0)
        task_label = torch.as_tensor(task_label, dtype=torch.long)

        # Padding
        if self.target_size is not None:
            pad = (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
            # image = F.pad(image, pad, mode=self.pad_mode, value=self.pad_value)
            measure = F.pad(measure, pad, mode=self.pad_mode, value=self.pad_value)

        return measure, task_label


class MRIDataset3(Dataset):
    def __init__(
            self,
            npy_dir_path,
            proportion: float = 1.0,
            use_padding: bool = True,
            target_size=(272, 320),
            pad_mode: str = "constant",
            pad_value: float = 0.0,
            verbose: bool = True,
    ):
        super().__init__()
        self.npy_dir_path = npy_dir_path
        self.proportion = proportion
        self.target_size = target_size if use_padding else None
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.verbose = verbose

        if self.verbose:
            print(f"[MRIDataset] Mapping data from {self.npy_dir_path}...")

        # ✅ LOAD IN INIT: This is safe for Multiprocessing with Numpy
        # The 'mmap_mode="r"' creates a lightweight pointer, not the full data.
        self.data_with_signal = np.load(os.path.join(self.npy_dir_path, "data.npy"), mmap_mode='r')
        self.invHg_data = np.load(os.path.join(self.npy_dir_path, "measure.npy"), mmap_mode='r')
        self.label = np.load(os.path.join(self.npy_dir_path, "label.npy"), mmap_mode='r')

        total_length = len(self.data_with_signal)
        _, self.H, self.W = self.data_with_signal.shape
        self.selected_length = int(total_length * self.proportion)

        # Pre-calculate padding
        self.pad_tuple = (0, 0, 0, 0)  # Left, Right, Top, Bottom
        if self.target_size is not None:
            target_H, target_W = self.target_size
            pad_h = target_H - self.H
            pad_w = target_W - self.W
            if pad_h < 0 or pad_w < 0:
                print(
                    f"Warning: Image size ({self.H},{self.W}) larger than target {self.target_size}. Cropping not implemented.")

            # PyTorch F.pad uses order: (Left, Right, Top, Bottom)
            self.pad_tuple = (
                pad_w // 2,  # Left
                pad_w - pad_w // 2,  # Right
                pad_h // 2,  # Top
                pad_h - pad_h // 2  # Bottom
            )

    def __len__(self):
        return self.selected_length

    def __getitem__(self, idx):
        # Indexing into mmap reads from disk (OS handles caching)
        # We use .copy() explicitly here to detach from the mmap backend immediately
        # into RAM, preventing any "buffer not writable" issues later.

        # Note: If your data is huge and you want to save RAM, remove .copy()
        # and rely on the .to(torch.float32) below to perform the copy.
        # 1. Fetch from mmap (Read-only view)
        image_view = self.data_with_signal[idx]
        measure_view = self.invHg_data[idx]
        label_view = self.label[idx]

        # 2. Force Copy to RAM (Now it is writeable)
        image_np = image_view.copy()
        measure_np = measure_view.copy()
        label_np = label_view.copy()

        # Convert to Tensor
        # torch.from_numpy() keeps the dtype.
        # .to(torch.float32) creates a new COPY in memory, which breaks the mmap link safely.
        image = torch.from_numpy(image_np).to(torch.float32).unsqueeze(0)
        measure = torch.from_numpy(measure_np).to(torch.float32).abs().unsqueeze(0)
        task_label = torch.as_tensor(label_np, dtype=torch.long)

        # Padding
        if self.target_size is not None:
            image = F.pad(image, self.pad_tuple, mode=self.pad_mode, value=self.pad_value)
            measure = F.pad(measure, self.pad_tuple, mode=self.pad_mode, value=self.pad_value)

        return image, measure, task_label


class MRIDataset3_2(Dataset):
    def __init__(
            self,
            npy_dir_path,
            proportion: float = 1.0,
            use_padding: bool = True,
            target_size=(272, 320),
            pad_mode: str = "constant",
            pad_value: float = 0.0,
            verbose: bool = True,
    ):
        super().__init__()
        self.npy_dir_path = npy_dir_path
        self.proportion = proportion
        self.target_size = target_size if use_padding else None
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.verbose = verbose

        if self.verbose:
            print(f"[MRIDataset] Mapping data from {self.npy_dir_path}...")

        # ✅ LOAD IN INIT: This is safe for Multiprocessing with Numpy
        # The 'mmap_mode="r"' creates a lightweight pointer, not the full data.
        # self.data_with_signal = np.load(os.path.join(self.npy_dir_path, "data.npy"), mmap_mode='r')
        self.invHg_data = np.load(os.path.join(self.npy_dir_path, "measure.npy"), mmap_mode='r')
        self.label = np.load(os.path.join(self.npy_dir_path, "label.npy"))

        total_length = len(self.label)
        _, self.H, self.W = self.invHg_data.shape
        self.selected_length = int(total_length * self.proportion)

        # Pre-calculate padding
        self.pad_tuple = (0, 0, 0, 0)  # Left, Right, Top, Bottom
        if self.target_size is not None:
            target_H, target_W = self.target_size
            pad_h = target_H - self.H
            pad_w = target_W - self.W
            if pad_h < 0 or pad_w < 0:
                print(
                    f"Warning: Image size ({self.H},{self.W}) larger than target {self.target_size}. Cropping not implemented.")

            # PyTorch F.pad uses order: (Left, Right, Top, Bottom)
            self.pad_tuple = (
                pad_w // 2,  # Left
                pad_w - pad_w // 2,  # Right
                pad_h // 2,  # Top
                pad_h - pad_h // 2  # Bottom
            )

    def __len__(self):
        return self.selected_length

    def __getitem__(self, idx):
        # Indexing into mmap reads from disk (OS handles caching)
        # We use .copy() explicitly here to detach from the mmap backend immediately
        # into RAM, preventing any "buffer not writable" issues later.

        # Note: If your data is huge and you want to save RAM, remove .copy()
        # and rely on the .to(torch.float32) below to perform the copy.
        # 1. Fetch from mmap (Read-only view)
        # image_view = self.data_with_signal[idx]
        measure_view = self.invHg_data[idx]
        label_view = self.label[idx]

        # 2. Force Copy to RAM (Now it is writeable)
        # image_np = image_view.copy()
        measure_np = measure_view.copy()
        label_np = label_view.copy()

        # Convert to Tensor
        # torch.from_numpy() keeps the dtype.
        # .to(torch.float32) creates a new COPY in memory, which breaks the mmap link safely.
        # image = torch.from_numpy(image_np).to(torch.float32).unsqueeze(0)
        measure = torch.from_numpy(measure_np).to(torch.float32).abs().unsqueeze(0)
        task_label = torch.as_tensor(label_np, dtype=torch.long)

        # Padding
        if self.target_size is not None:
            # image = F.pad(image, self.pad_tuple, mode=self.pad_mode, value=self.pad_value)
            measure = F.pad(measure, self.pad_tuple, mode=self.pad_mode, value=self.pad_value)

        return measure, task_label


class MRIDataset(Dataset):
    def __init__(
        self,
        h5_file_path,
        proportion: float = 1.0,
        use_padding: bool = True,
        target_size=(272, 320),   # Unified input dimensions
        pad_mode: str = "constant",
        pad_value: float = 0.0,
    ):
        """
        h5_file_path: Path to the generated data.h5 file.
        proportion:   Percentage of the dataset to use (1.0 = 100%).
        use_padding:  Whether to pad 260x311 data to the target_size.
        target_size:  (target_H, target_W), currently set to (272, 320).
        pad_mode:     Padding style: "constant", "reflect", or "replicate".
        pad_value:    The value used for filling when using "constant" mode.
        """
        super().__init__()

        # Lazy loading initialization
        self.h5_file = h5py.File(h5_file_path, "r")

        # Calculate dataset length
        self.total_length = int(len(self.h5_file["data_with_signal"]))
        self.selected_length = int(self.total_length * proportion)

        # Store references only; do not load data into memory yet
        self.data_with_signal_ds = self.h5_file["data_with_signal"]  # (N, H, W)
        self.invHg_data_ds = self.h5_file["invHg_data"]              # (N, H, W)
        self.label_ds = self.h5_file["label"]                        # (N,)

        # Original dimensions (expected: 260 x 311)
        _, self.H, self.W = self.data_with_signal_ds.shape
        print(f"[MRIDataset] original size: ({self.H}, {self.W})")

        self.use_padding = use_padding
        self.target_size = target_size if use_padding else None
        self.pad_mode = pad_mode
        self.pad_value = pad_value

        if self.target_size is not None:
            target_H, target_W = self.target_size
            assert target_H >= self.H and target_W >= self.W, \
                f"target_size {self.target_size} must be >= original size ({self.H}, {self.W})"

            # Calculate symmetric/approximate padding
            pad_h_total = target_H - self.H   # 272 - 260 = 12
            pad_w_total = target_W - self.W   # 320 - 311 = 9

            self.pad_top = pad_h_total // 2   # 6
            self.pad_bottom = pad_h_total - self.pad_top  # 6
            self.pad_left = pad_w_total // 2  # 4
            self.pad_right = pad_w_total - self.pad_left  # 5

            print(
                f"[MRIDataset] will pad from ({self.H}, {self.W}) "
                f"to ({target_H}, {target_W}) with "
                f"(top={self.pad_top}, bottom={self.pad_bottom}, "
                f"left={self.pad_left}, right={self.pad_right}), "
                f"mode={self.pad_mode}"
            )
        else:
            self.pad_top = self.pad_bottom = self.pad_left = self.pad_right = 0
            print(f"[MRIDataset] no padding, using original size ({self.H}, {self.W})")

        print("dataset size:", self.selected_length)
        print("True lazy loading: No data loaded into memory yet")

    def __len__(self):
        return self.selected_length

    def __getitem__(self, idx):
        # Lazy load a single sample from h5
        image = self.data_with_signal_ds[idx]    # (H, W)
        measure = self.invHg_data_ds[idx]        # (H, W)
        task_label = self.label_ds[idx]          # 标量

        # Convert to tensor: [H, W] -> [1, H, W]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        # Assuming invHg_data is already stored as absolute values; if not, apply abs()
        measure = torch.tensor(np.abs(measure), dtype=torch.float32).unsqueeze(0)
        task_label = torch.tensor(task_label, dtype=torch.long)

        # Apply padding (if enabled)
        if self.target_size is not None:
            # F.pad sequence: (left, right, top, bottom)
            pad = (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
            image = F.pad(image, pad, mode=self.pad_mode, value=self.pad_value)
            measure = F.pad(measure, pad, mode=self.pad_mode, value=self.pad_value)

        return image, measure, task_label

    def close(self):
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()
            print("HDF5 file closed")

    def __del__(self):
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()

import os
import torch.nn.functional as F
from torch.utils.data import Dataset


class MRIDataset4(Dataset):
    def __init__(
            self,
            npy_dir_path,  # Point to the folder containing .npy files
            proportion: float = 1.0,
            use_padding: bool = True,
            target_size=(272, 320),
            pad_mode: str = "constant",
            pad_value: float = 0.0,
            verbose: bool = True,
    ):
        super().__init__()
        self.npy_dir_path = npy_dir_path
        self.proportion = proportion
        self.target_size = target_size if use_padding else None
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.verbose = verbose

        if self.verbose:
            print(f"[MRIDatasetMmap] Mapping data from {self.npy_dir_path}...")

        # ✅ 1) Load with mmap_mode='r' (Read-only)
        # This acts like a pointer. It does NOT read data into RAM yet.
        self.invHg_data = np.load(os.path.join(self.npy_dir_path, "measure.npy"), mmap_mode='r')
        self.label = np.load(os.path.join(self.npy_dir_path, "label.npy"), mmap_mode='r')

        # Optional: Load image data if needed later
        # self.data_with_signal = np.load(os.path.join(self.npy_dir_path, "data.npy"), mmap_mode='r')

        # Get shapes (Metadata is read instantly from the npy header)
        self.total_length = len(self.invHg_data)
        # Assuming shape is (N, H, W) -> take H, W from the first sample logic
        _, self.H, self.W = self.invHg_data.shape

        self.selected_length = int(self.total_length * self.proportion)

        # ✅ 2) Pre-calculate pad parameters
        self.pad_tuple = (0, 0, 0, 0)
        if self.target_size is not None:
            target_H, target_W = self.target_size
            pad_h = target_H - self.H
            pad_w = target_W - self.W

            # PyTorch F.pad order: (Left, Right, Top, Bottom)
            self.pad_tuple = (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2
            )

    def __len__(self):
        return self.selected_length

    def __getitem__(self, idx):
        # (Optional) Ensure idx is within bounds
        if idx < 0 or idx >= self.selected_length:
            raise IndexError

        # ✅ 3) Fetch and Copy
        # We perform an explicit .copy() here.
        # This forces the OS to read the disk page into a new WRITEABLE memory block.
        # This prevents PyTorch warnings about "negative strides" or "read-only tensors".
        measure_np = self.invHg_data[idx].copy()
        label_np = self.label[idx].copy()

        # Convert to Tensor
        measure = torch.from_numpy(measure_np).to(torch.float32).abs().unsqueeze(0)
        task_label = torch.as_tensor(label_np, dtype=torch.long)

        # Padding (Uses pre-calculated tuple for speed)
        if self.target_size is not None:
            measure = F.pad(measure, self.pad_tuple, mode=self.pad_mode, value=self.pad_value)

        return measure, task_label




# class MRIDataset(Dataset):
#     def __init__(self, h5_file_path, proportion=1.0):
#         assert 0 < proportion <= 1.0
#         self.h5_path = h5_file_path
#         self.proportion = float(proportion)

#         # 只读一次元信息，立刻关闭，避免把已打开句柄带到子进程
#         with h5py.File(self.h5_path, 'r') as f:
#             total = len(f['data_with_signal'])

#         self.total_length = int(total)
#         self.selected_length = int(self.total_length * self.proportion)

#         # 这些在每个进程里延迟打开
#         self._h5 = None
#         self.data_with_signal_ds = None
#         self.invHg_data_ds = None
#         self.label_ds = None

#         print("dataset size:", self.selected_length)
#         print("Lazy loading per-worker: file will be opened on first access")

#     def __len__(self):
#         return self.selected_length

#     def _ensure_open(self):
#         """在当前进程第一次访问时打开 HDF5 文件（只读、SWMR 更稳）"""
#         if self._h5 is None:
#             # swmr=True 适合多进程只读；libver='latest' 常与 swmr 配合
#             self._h5 = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')
#             self.data_with_signal_ds = self._h5['data_with_signal']
#             self.invHg_data_ds      = self._h5['invHg_data']
#             self.label_ds           = self._h5['label']

#     def __getitem__(self, idx):
#         # 注意：DataLoader 会只给 0..selected_length-1 的 idx，这里直接用即可
#         self._ensure_open()

#         image = self.data_with_signal_ds[idx]   # [H, W] 或 [H, W, ...]
#         measure = self.invHg_data_ds[idx]
#         task_label = self.label_ds[idx] - 1     # 确保从 0 开始

#         # 转 tensor（放在 CPU；到 GPU 的搬运放到训练循环里）
#         image   = torch.tensor(image, dtype=torch.float32).unsqueeze(0)         # -> [1, H, W]
#         measure = torch.tensor(np.abs(measure), dtype=torch.float32).unsqueeze(0)
#         task_label = torch.tensor(task_label, dtype=torch.long)

#         return image, measure, task_label

#     def close(self):
#         if self._h5 is not None:
#             try:
#                 self._h5.close()
#             except Exception:
#                 pass
#             finally:
#                 self._h5 = None
#                 self.data_with_signal_ds = None
#                 self.invHg_data_ds = None
#                 self.label_ds = None

#     def __del__(self):
#         self.close()

#     def __getstate__(self):
#         """防止带着已打开的句柄被多进程复制（pickle）"""
#         state = self.__dict__.copy()
#         state['_h5'] = None
#         state['data_with_signal_ds'] = None
#         state['invHg_data_ds'] = None
#         state['label_ds'] = None
#         return state
    
# class MRIDataset(Dataset):
#     def __init__(self, h5_file_path, proportion=1.0):
#         self.h5_file = h5py.File(h5_file_path, 'r')  # Lazy loading
        
#         # Data length calculation
#         self.total_length = int(len(self.h5_file['data_with_signal']))
#         self.selected_length = int(self.total_length * proportion)

#         # ✅ 修改这里：只保存数据集引用，而不是立即加载数据
#         self.data_with_signal_ds = self.h5_file['data_with_signal']  # 只是引用
#         self.invHg_data_ds = self.h5_file['invHg_data']              # 只是引用
#         self.label_ds = self.h5_file['label']                        # 只是引用

#         print("dataset size:", self.selected_length)
#         print("True lazy loading: No data loaded into memory yet")

#     def __len__(self):
#         return self.selected_length

#     def __getitem__(self, idx):
#         # ✅ 修改这里：从数据集引用中按需加载单个样本
#         image = self.data_with_signal_ds[idx]    # 只加载一个样本
#         measure = self.invHg_data_ds[idx]        # 只加载一个样本
#         task_label = self.label_ds[idx]    # Adjust labels to start from 0

#         # Convert to Torch Tensor
#         image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [H, W] -> [1, H, W]
#         measure = torch.tensor(np.abs(measure), dtype=torch.float32).unsqueeze(0)
#         task_label = torch.tensor(task_label, dtype=torch.long)

#         return image, measure, task_label

#     def close(self):
#         """手动关闭HDF5文件"""
#         if hasattr(self, 'h5_file') and self.h5_file:
#             self.h5_file.close()
#             print("HDF5 file closed")

#     def __del__(self):
#         """确保文件被关闭"""
#         if hasattr(self, 'h5_file') and self.h5_file:
#             self.h5_file.close()


class MRIDataset_test(Dataset):
    def __init__(self, h5_file_path, proportion=1.0):
        self.h5_file = h5py.File(h5_file_path, 'r')  # Lazy loading
        
        # Data length calculation
        self.total_length = int(len(self.h5_file['data_with_signal']))
        self.selected_length = int(self.total_length * proportion)

        # ✅ 修改这里：只保存数据集引用，而不是立即加载数据
        self.data_with_signal_ds = self.h5_file['data_with_signal']  # 只是引用
        self.invHg_data_ds = self.h5_file['invHg_data']              # 只是引用
        self.label_ds = self.h5_file['label']                        # 只是引用

        print("dataset size:", self.selected_length)
        print("True lazy loading: No data loaded into memory yet")

    def __len__(self):
        return self.selected_length

    def __getitem__(self, idx):
        # ✅ 修改这里：从数据集引用中按需加载单个样本
        image = self.data_with_signal_ds[idx]    # 只加载一个样本
        measure = self.invHg_data_ds[idx]        # 只加载一个样本
        task_label = self.label_ds[idx] - 1      # Adjust labels to start from 0

        # Convert to Torch Tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [H, W] -> [1, H, W]
        measure = torch.tensor(np.abs(measure), dtype=torch.float32).unsqueeze(0)
        task_label = torch.tensor(task_label, dtype=torch.long)

        return image, measure, task_label

    def close(self):
        """手动关闭HDF5文件"""
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()
            print("HDF5 file closed")

    def __del__(self):
        """确保文件被关闭"""
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()


# class MRIDataset(Dataset):
#     def __init__(self, h5_file_path, proportion=1.0):
#         self.h5_file = h5py.File(h5_file_path, 'r')  # Lazy loading
#         # Data length calculation
#         self.total_length = int(len(self.h5_file['data_with_signal']))
#         selected_length = int(self.total_length * proportion)

#         self.data_with_signal = self.h5_file['data_with_signal'][:selected_length]
#         self.invHg_data = self.h5_file['invHg_data'][:selected_length]
#         self.label = self.h5_file['label'][:selected_length]
#         print("dataset size:", len(self.data_with_signal))

#     def __len__(self):
#         return len(self.data_with_signal)

#     def __getitem__(self, idx):
#         # Load data on demand
#         image = self.h5_file['data_with_signal'][idx]
#         measure = self.h5_file['invHg_data'][idx]
#         task_label = self.h5_file['label'][idx] - 1  # Adjust labels to start from 0

#         # Convert to Torch Tensor
#         image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [H, W] -> [1, H, W]
#         measure = torch.tensor(np.abs(measure), dtype=torch.float32).unsqueeze(0)
#         task_label = torch.tensor(task_label, dtype=torch.long)

#         return image, measure, task_label

# --- changjie version-----
# class LumpyDataset(Dataset):
#     def __init__(self, h5_file_path, proportion=1.0):
#         self.h5_file = h5py.File(h5_file_path, 'r')  # Lazy loading
#         # Data length calculation
#         self.total_length = int(len(self.h5_file['data_with_signal']))
#         selected_length = int(self.total_length * proportion)

#         self.data_with_signal = self.h5_file['data_with_signal'][:selected_length]
#         self.label = self.h5_file['labels'][:selected_length]
#         print("dataset size:", len(self.data_with_signal))

#     def __len__(self):
#         return len(self.data_with_signal)

#     def __getitem__(self, idx):
#         # Load data on demand
#         image = self.h5_file['data_with_signal'][idx]
#         task_label = self.h5_file['labels'][idx]

#         # Convert to Torch Tensor
#         image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [H, W] -> [1, H, W]
#         task_label = torch.tensor(task_label, dtype=torch.long)

#         return image, task_label



class LumpyDataset(Dataset):
    def __init__(self, h5_file_path, proportion=1.0, shuffle=False, seed=0):
        """
        真正懒加载版本：
        - 初始化阶段不把数据读入内存，只记录文件路径与采样索引
        - 每次 __getitem__ 时从 HDF5 读取单条样本
        - 兼容多进程 DataLoader（worker 内独立打开文件）
        """
        assert 0 < proportion <= 1.0
        self.h5_file_path = h5_file_path
        self.h5_file = None  # 延后到第一次 __getitem__ 在各自进程中打开

        # 先临时开一次文件拿长度，然后立即关掉（避免持有句柄被 DataLoader fork）
        with h5py.File(self.h5_file_path, 'r') as f:
            total_len = int(len(f['data_with_signal']))

        # 只保存索引，不切片数据到内存
        sel_len = int(total_len * proportion)
        self.indices = np.arange(total_len, dtype=np.int64)[:sel_len]
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)

        print(f"dataset size: {len(self.indices)}")
        print("True lazy loading: no dataset is loaded into RAM at init.")

    def _ensure_open(self):
        """
        确保当前（主进程或各 worker 进程）里已打开 HDF5。
        DataLoader 多进程下，每个 worker 都会各自执行一遍 __getitem__，
        因此不能在主进程里把 h5 句柄传给子进程。
        """
        if self.h5_file is None:
            # 只读打开；不要使用 swmr=True（没必要且更慢）
            self.h5_file = h5py.File(self.h5_file_path, 'r')
            # 可缓存关键 dataset 引用（仍然是懒加载）
            self._img_ds = self.h5_file['data_with_signal']
            self._lbl_ds = self.h5_file['labels']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        self._ensure_open()
        idx = int(self.indices[i])

        # 按需读取单条样本（仍在磁盘，不进内存切片）
        image = self._img_ds[idx]      # shape: [H, W]
        label = self._lbl_ds[idx]      # 标量（0/1 或 1/2）

        # 转成张量（单通道 [1, H, W]）
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        # 如需把 1/2 改为 0/1，可在此减 1；若本来就是 0/1 则不要减
        task_label = torch.tensor(int(label), dtype=torch.long)

        return image, task_label

    def close(self):
        """手动关闭 HDF5 文件句柄（主进程或各 worker 各自关闭）"""
        if getattr(self, 'h5_file', None) is not None:
            try:
                self.h5_file.close()
            finally:
                self.h5_file = None

    def __del__(self):
        # 防止对象销毁时句柄未关闭
        try:
            self.close()
        except Exception:
            pass




import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class LumpyDatasetEstimation(Dataset):
    """
    懒加载 + 多进程安全的数据集，始终返回三元组：
        image: FloatTensor [1, H, W]
        label: LongTensor  标量（0/1）
        rs   : FloatTensor [2]（若无则返回 [0., 0.] 占位；对 SA 样本可为 0 或 NaN，训练时会被 mask 掉）
    """
    def __init__(
        self,
        h5_file_path: str,
        proportion: float = 1.0,
        shuffle: bool = False,
        seed: int = 0,
        one_based_labels: bool = False,   # 若你的标签存为 {1,2} ，置 True 转成 {0,1}
        rs_dim: int = 2,                  # 估计参数维度（默认 rs=(x,y)）
        return_nan_for_absent: bool = False,  # 若 H5 无 rs 时，用 NaN 占位而非 0
    ):
        assert 0 < proportion <= 1.0
        self.h5_file_path = h5_file_path
        self.h5_file = None  # 延迟到 __getitem__ 内部在各自 worker 打开

        self.one_based_labels = one_based_labels
        self.rs_dim = rs_dim
        self.return_nan_for_absent = return_nan_for_absent

        # 预探测：确定数据集键名与长度
        with h5py.File(self.h5_file_path, 'r') as f:
            if 'data_with_signal' in f:
                self._data_key = 'data_with_signal'
            elif 'data' in f:
                self._data_key = 'data'
            else:
                raise KeyError("H5 文件缺少数据集：'data_with_signal' 或 'data'。")

            if 'labels' not in f:
                raise KeyError("H5 文件缺少数据集：'labels'。")
            self._label_key = 'labels'

            self._has_rs = 'rs' in f  # 是否包含估计真值（如 rs=(x,y)）
            total_len = int(len(f[self._data_key]))

        # 子采样索引（不把数据切片到内存）
        sel_len = int(total_len * proportion)
        self.indices = np.arange(total_len, dtype=np.int64)[:sel_len]
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)

        print(f"[LumpyDataset] file={h5_file_path}")
        print(f"[LumpyDataset] keys: data='{self._data_key}', labels='{self._label_key}', rs_exists={self._has_rs}")
        print(f"[LumpyDataset] size: {len(self.indices)} / total {total_len}  (lazy-load)")

    def _ensure_open(self):
        """在当前进程/worker 内确保已打开 HDF5 句柄，并缓存 dataset 引用。"""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_file_path, 'r')  # 只读
            self._img_ds = self.h5_file[self._data_key]       # [N,H,W]
            self._lbl_ds = self.h5_file[self._label_key]      # [N]
            self._rs_ds  = self.h5_file['rs'] if self._has_rs else None  # [N,rs_dim]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        self._ensure_open()
        idx = int(self.indices[i])

        # 读取
        image = self._img_ds[idx]   # [H, W]
        label = int(self._lbl_ds[idx])

        # 标签转 0/1（若为 1/2）
        if self.one_based_labels:
            label = label - 1

        # 张量化
        image_t = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1,H,W]
        label_t = torch.tensor(label, dtype=torch.long)

        # rs 读取/占位
        if self._has_rs:
            rs = self._rs_ds[idx]  # 形如 [x, y]；SA 样本可为 0 或 NaN
            rs_t = torch.tensor(rs, dtype=torch.float32)
        else:
            if self.return_nan_for_absent:
                rs_fill = np.full((self.rs_dim,), np.nan, dtype=np.float32)
            else:
                rs_fill = np.zeros((self.rs_dim,), dtype=np.float32)
            rs_t = torch.tensor(rs_fill, dtype=torch.float32)

        return image_t, label_t, rs_t

    def close(self):
        """手动关闭 HDF5 句柄（主进程或各 worker 各自关闭）"""
        if getattr(self, 'h5_file', None) is not None:
            try:
                self.h5_file.close()
            finally:
                self.h5_file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass



if __name__ == '__main__':
    # Data path configuration
    filename = "sks_3_0.04_15_c2_num_signals"
    base_dir = f'/shared/anastasio-s2/SI/HCP_selected/{filename}'
    train_h5_path = os.path.join(base_dir, 'val', 'data.h5')

    # Create dataset and dataloader
    train_dataset = MRIDataset(train_h5_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,  # Improve GPU data loading efficiency
        prefetch_factor=8,  # Amount of data preloaded per thread
        persistent_workers=True  # Keep workers alive, improve multi-epoch performance
    )

    # # Test speed
    # i = 0
    start = time.time()
    zero = 0
    one = 0
    for image, measure, task_label in train_dataloader:
        # count zero and one in task_label
        task_label = task_label.numpy()
        zero_count = np.sum(task_label == 0)
        one_count = np.sum(task_label == 1)
        zero += zero_count
        one += one_count
    
    print("zero:", zero)
    print("one:", one)

    # train_dataset = LumpyDataset(train_h5_path)
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=32,
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True,  # Improve GPU data loading efficiency
    #     prefetch_factor=8,  # Amount of data preloaded per thread
    #     persistent_workers=True  # Keep workers alive, improve multi-epoch performance
    # )