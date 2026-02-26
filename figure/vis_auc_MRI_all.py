import matplotlib.pyplot as plt
import os

# ========================
# 1. Data
# ========================
data_proportion = [0.005, 0.5, 1.0]
data_sample_numbers = ['1.7K', '17.2K', '34.3K']
# data_proportion = [0.5, 1.0]

## AUC
# SKE
auc_cnn_io = [0.83142, 0.91277, 0.91543]
# auc_vib_ce = [0.87018, 0.91432, 0.91701]
auc_vib_io = [0.87255, 0.91440, 0.91705]

# SKS
# auc_cnn_io = [0.91331, 0.94238]
# auc_vib_ce = [0.91504, 0.94242]
# auc_vib_io = [0.91115, 0.9466]  # 0.95106 94551


## Std
# SKE
std_cnn_io = [0.00388, 0.00271, 0.00262]
# std_vib_ce = [0.00340, 0.00265, 0.00259]
std_vib_io = [0.00330, 0.00265, 0.00261]

# SKS
# std_cnn_io = [0.00310, 0.00243]
# std_vib_ce = [0.00296, 0.00250]
# std_vib_io = [0.00318, 0.00235]  # 0.00229


# ========================
# 2. Save path
# ========================
save_dir = "/home/chunsup2/PycharmProjects/VIBIO/figure/"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "mri_data_proportion_auc_ske.png")

# ========================
# 3. Plot
# ========================
# plt.figure(figsize=(6, 4))
#
# plt.errorbar(
#     data_proportion, auc_cnn_io, yerr=std_cnn_io,
#     fmt='-o', capsize=5, linewidth=0.8, markersize=2,
#     label="CNN-IO"
# )
#
# plt.errorbar(
#     data_proportion, auc_vib_ce, yerr=std_vib_ce,
#     fmt='-s', capsize=5, linewidth=0.8, markersize=2,
#     label="VIB-CE"
# )
#
# plt.errorbar(
#     data_proportion, auc_vib_io, yerr=std_vib_io,
#     fmt='-^', capsize=5, linewidth=0.8, markersize=2,
#     label="VIB-IO"
# )

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 15,
    'axes.titlesize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 12,
    'grid.alpha': 0.5,
})

plt.figure(figsize=(8, 6))

# Colors
color_cnn = '#003366'  # Navy Blue
color_ce  = '#D35400'  # Burnt Orange
color_io  = '#004d00'  # Forest Green

# CNN-IO (Blue)

plt.errorbar(
    data_proportion, auc_cnn_io, yerr=std_cnn_io,
    fmt='-o', linestyle='dashdot', capsize=6, linewidth=2.0, markersize=8,
    color=color_cnn, ecolor=color_cnn, elinewidth=1.5,
    label="CNN-IO", zorder=3
)

# VIB-CE (Orange)
# plt.errorbar(
#     data_proportion, auc_vib_ce, yerr=std_vib_ce,
#     fmt='-s', linestyle='dashdot', capsize=6, linewidth=2.0, markersize=8,
#     color=color_ce, ecolor=color_ce, elinewidth=1.5,
#     label="VIB-CE", zorder=3
# )

# VIB-IO (Green)
plt.errorbar(
    data_proportion, auc_vib_io, yerr=std_vib_io,
    fmt='-^', linestyle='dashdot', capsize=6, linewidth=2.0, markersize=8,
    color=color_io, ecolor=color_io, elinewidth=1.5,
    label="VIB-IO", zorder=3
)


# ========================
# 4. Labels & style
# ========================
plt.xlabel("# Training Samples", fontsize=21)
plt.ylabel("AUC", fontsize=21)

plt.xticks(data_proportion, data_sample_numbers, fontsize=15)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=18, loc='lower right')

plt.margins(y=0.15)
plt.tight_layout()

# ========================
# 5. Save & show
# ========================
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Save path：{save_path}")
