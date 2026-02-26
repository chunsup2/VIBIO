import matplotlib.pyplot as plt
import os

# ========================
# 1. Data
# ========================
data_proportion = [0.005, 0.5, 1.0]

# --- Best (Optional/Reference) --- D=4
# auc_cnn_io = [0.51986, 0.91764, 0.93724]
# auc_vib_ce = [0.53071, 0.89331, 0.93756]
# auc_vib_io = [0.75278, 0.87570, 0.88106]

# auc_cnn_io = [0.51872, 0.91720, 0.93233]
# auc_vib_ce = [0.53071, 0.89331, 0.93756]
# auc_vib_io = [0.75278, 0.87570, 0.88106]


### Best Loss
# AUC
# auc_cnn_io = [0.50515, 0.90734, 0.93530]
# auc_vib_ce = [0.50562, 0.90832, 0.93865]
# auc_vib_io = [0.50474, 0.90759, 0.94551]

# Standard Deviation (Set to 0 if not available, or remove errorbar)
# std_cnn_io = [0.00376, 0.00322, 0.00264]
# std_vib_ce = [0.00425, 0.00320, 0.00261]
# std_vib_io = [0.00353, 0.00322, 0.00241]

### Best AUC
# AUC
auc_cnn_io = [0.50553, 0.91331, 0.94238]
auc_vib_ce = [0.50686, 0.91504, 0.94242]
auc_vib_io = [0.50758, 0.91115, 0.9474]

# Standard Deviation (Set to 0 if not available, or remove errorbar)
std_cnn_io = [0.00410, 0.00310, 0.00243]
std_vib_ce = [0.00473, 0.00296, 0.00250]
std_vib_io = [0.00479, 0.00318, 0.00200]

# ========================
# 2. Save path
# ========================
save_dir = "/home/chunsup2/PycharmProjects/IOVIB/figure/"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "mri_data_proportion_auc_sks.png")

# ========================
# 3. Plot
# ========================
plt.figure(figsize=(6, 4))

plt.errorbar(
    data_proportion, auc_cnn_io, yerr=std_cnn_io,
    fmt='-o', capsize=5, linewidth=0.8, markersize=2,
    label="CNN-IO"
)

plt.errorbar(
    data_proportion, auc_vib_ce, yerr=std_vib_ce,
    fmt='-s', capsize=5, linewidth=0.8, markersize=2,
    label="VIB-CE"
)

plt.errorbar(
    data_proportion, auc_vib_io, yerr=std_vib_io,
    fmt='-^', capsize=5, linewidth=0.8, markersize=2,
    label="VIB-IO"
)

# Annotations (New Code)
# ========================
# Loop through each data point to add text
for i, x in enumerate(data_proportion):
    y_cnn = auc_cnn_io[i]
    y_ce = auc_vib_ce[i]
    y_io = auc_vib_io[i]

    # Format string to 3 or 4 decimal places
    fmt = "{:.4f}"

    # CNN-IO (Blue): Place text slightly below
    plt.text(x, y_cnn - 0.025, fmt.format(y_cnn),
             ha='center', va='top', fontsize=9, color='tab:blue', fontweight='bold')

    # VIB-CE (Orange): Adjust position based on value relative to others
    # Using a slight right offset for some points to avoid clutter
    if i == 1: # Middle point is highest
        plt.text(x, y_ce + 0.015, fmt.format(y_ce),
                 ha='center', va='bottom', fontsize=9, color='tab:orange', fontweight='bold')
    else:
        plt.text(x + 0.02, y_ce, fmt.format(y_ce),
                 ha='left', va='center', fontsize=9, color='tab:orange', fontweight='bold')

    # VIB-IO (Green): Adjust position
    if i == 1: # Middle point is lowest
        plt.text(x, y_io - 0.045, fmt.format(y_io),
                 ha='center', va='top', fontsize=9, color='tab:green', fontweight='bold')
    else:
        plt.text(x, y_io + 0.015, fmt.format(y_io),
                 ha='center', va='bottom', fontsize=9, color='tab:green', fontweight='bold')

# ========================
# 4. Labels & style
# ========================
plt.xlabel("Data Proportion", fontsize=12)
plt.ylabel("AUC", fontsize=12)
plt.xticks(data_proportion)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.margins(y=0.15)
plt.tight_layout()

# ========================
# 5. Save & show
# ========================
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Save path：{save_path}")
