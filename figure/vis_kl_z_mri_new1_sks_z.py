import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ===================== 1) Manual Data Entry =====================
manual_data = [
    # --- z = 256 ---
    {"z": 256, "kl": 100.0, "auc": 0.8747},
    {"z": 256, "kl": 1.0, "auc": 0.8872},
    {"z": 256, "kl": 0.1, "auc": 0.9293},
    {"z": 256, "kl": 0.01, "auc": 0.9303},
    {"z": 256, "kl": 0.001, "auc": 0.9458},
    {"z": 256, "kl": 0.0001, "auc": 0.9326},
    {"z": 256, "kl": 1e-05, "auc": 0.8924},
    {"z": 256, "kl": 1e-08, "auc": 0.8474},

    # --- z = 512 ---
    {"z": 512, "kl": 100.0, "auc": 0.8347},
    {"z": 512, "kl": 1.0, "auc": 0.9182},
    {"z": 512, "kl": 0.1, "auc": 0.9452},
    {"z": 512, "kl": 0.01, "auc": 0.9474},
    {"z": 512, "kl": 0.001, "auc": 0.9463},
    {"z": 512, "kl": 0.0001, "auc": 0.9442},
    {"z": 512, "kl": 1e-05, "auc": 0.9455},
    {"z": 512, "kl": 1e-08, "auc": 0.8654},

    # --- z = 1024 ---
    {"z": 1024, "kl": 100.0, "auc": 0.8195},
    {"z": 1024, "kl": 1.0, "auc": 0.8898},
    {"z": 1024, "kl": 0.1, "auc": 0.9147},
    {"z": 1024, "kl": 0.01, "auc": 0.9311},
    {"z": 1024, "kl": 0.001, "auc": 0.9421},
    {"z": 1024, "kl": 0.0001, "auc": 0.9466},
    {"z": 1024, "kl": 1e-05, "auc": 0.9445},
    {"z": 1024, "kl": 1e-08, "auc": 0.8724},
]

# ===================== 2) Process Data =====================
df = pd.DataFrame(manual_data)
if df.empty:
    raise ValueError("No data found.")

# Sort to ensure lines connect properly
df = df.sort_values(["kl", "z"]).reset_index(drop=True)

# ===================== 3) Plotting =====================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 15,
    'axes.titlesize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 11,
    'grid.alpha': 0.5,
    'lines.linewidth': 2.0,
    'lines.markersize': 9
})

plt.figure(figsize=(9, 7))

# Get unique Beta values
unique_kls = sorted(df["kl"].unique(), reverse=True)

# === Distinct Colors ===
distinct_colors = [
    '#E6194B', '#3CB44B', '#FFE119', '#4363D8',
    '#F58231', '#911EB4', '#42D4F4', '#F032E6', '#000000'
]

# === Distinct Markers ===
distinct_markers = ['o', 's', '^', 'D', 'v', 'X', '*', 'P']

for i, kl_val in enumerate(unique_kls):
    # Filter data for this specific Beta
    g = df[df["kl"] == kl_val].sort_values("z")

    c = distinct_colors[i % len(distinct_colors)]
    m = distinct_markers[i % len(distinct_markers)]

    # --- FIXED LABEL FORMATTING ---
    # We now enclose the entire string in $...$ to ensure it is parsed as a single math expression.
    if kl_val >= 100:
        label_str = f"$\\beta={int(kl_val)}$"
    elif kl_val == 1.0:
        label_str = r"$\beta=1.0$"
    elif kl_val < 0.0001:
        # Scientific notation inside math mode
        label_str = f"$\\beta={kl_val:.0e}$"
    else:
        label_str = f"$\\beta={kl_val}$"

    plt.plot(g["z"], g["auc"], marker=m, linestyle='--', color=c, label=label_str, linewidth=2.5)

# Labels & Title
plt.xlabel(r"Latent Dimension ($z$)", fontsize=21)
plt.ylabel("AUC", fontsize=21)
plt.title(r"AUC vs $z$ (Fixed $\beta$)", fontsize=21)
plt.grid(True, which="both", linestyle="--", linewidth=0.6)

# X-Axis Scaling
plt.xscale("log", base=2)
unique_zs = sorted(df["z"].unique())
plt.xticks(unique_zs, [str(int(z)) for z in unique_zs])
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())

# Legend placement
plt.legend(title=r'KL Weight ($\beta$)', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.tight_layout()

# ===================== 4) Save =====================
save_dir = "/home/chunsup2/PycharmProjects/VIBIO/figure/"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "auc_vs_z_fixed_beta_manual.png")

plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {save_path}")

plt.show()