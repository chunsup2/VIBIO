

#   --------------- MRI -------------------

import numpy as np
import matplotlib.pyplot as plt

# ================== Global Style Settings (Adjust as needed) ==================
# plt.rcParams["font.family"] = "Times New Roman"   # Standard paper font; can change to SimHei etc.
# plt.rcParams["font.size"] = 12
# plt.rcParams["axes.linewidth"] = 1.0

# ================== Your Values ==================
auc_std_dict = {
    "CNN-IO":  (0.94238, 0.00243),
    "VIB-CE":  (0.94242, 0.00250),
    "VIB-IO":  (0.94660, 0.00235),
}

files = {
    # "MCMC-IO":  "/home/chunsup2/PycharmProjects/VIBIO/figure/roc_chicago/point_lumpy_mcmc.txt",
    "CNN-IO":   "/home/chunsup2/PycharmProjects/VIBIO/figure/roc_chicago/point_sksbks_mri_cnnio.txt",
    "VIB-CE": "/home/chunsup2/PycharmProjects/VIBIO/figure/roc_chicago/point_sksbks_mri_vibce.txt",
    "VIB-IO": "/home/chunsup2/PycharmProjects/VIBIO/figure/roc_chicago/point_sksbks_mri_vibio.txt"
}


# Colors and linestyles: Mimicking your reference diagram as closely as possible
colors = {
    # "MCMC-IO": "tab:blue",
    "CNN-IO": "tab:blue",
    "VIB-CE": "tab:orange",
    "VIB-IO": "tab:green"
}

linestyles = {
    # "MCMC-IO": "-",
    "CNN-IO": "--",
    "VIB-CE": "--",
    "VIB-IO": "--"
}

linewidths = {
    # "MCMC-IO": 3.0,
    "CNN-IO": 3.0,
    "VIB-CE": 3.0,
    "VIB-IO": 3.0
}

# ================== Plotting ==================
plt.figure(figsize=(5,4.5))   # Proportions close to standard academic figures

for method, filename in files.items():
    data = np.loadtxt(filename)
    fpr, tpr = data[:, 0], data[:, 1]

    auc, std = auc_std_dict[method]
    # Legend text format: AUC_Method = 0.799 ± 0.007
    label = f"{method} = {auc}±{std}"

    plt.plot(
        fpr, tpr,
        label=label,
        color=colors[method],
        linestyle=linestyles[method],
        linewidth=linewidths[method]
    )

# Axis labels: TPF / FPF
plt.xlabel("FPF", fontsize=18)   # False Positive Fraction
plt.ylabel("TPF", fontsize=18)   # True Positive Fraction

# Set axes from 0 to 1
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Disable grid lines to keep the plot clean
plt.grid(False)

# Place legend in the bottom right corner (inside the plot)
plt.legend(
    loc="lower right",
    frameon=True,
    framealpha=1.0,
    facecolor="white",
    edgecolor="black",
    fontsize=12
)

plt.tight_layout()

save_path = "/home/chunsup2/PycharmProjects/VIBIO/figure/roc_sksbks_mri.png"
plt.savefig(save_path, dpi=300)
# You can also save as PDF, which is better for publications:
plt.show()

print("Image saved as: roc_sksbks_mri.png")

