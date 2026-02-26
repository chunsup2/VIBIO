import os
import pandas as pd
import matplotlib.pyplot as plt

# ===================== 1) Manual Data Entry =====================
# Add your data points here as dictionaries.
# 'z': Latent dimension size
# 'kl': Beta (KL weight)
# 'auc': The AUC score
manual_data = [
    # # --- z = 128 ---
    # {"z": 128, "kl": 100.0, "auc": 0.5217},
    # {"z": 128, "kl": 1.0, "auc": 0.5053},
    # {"z": 128, "kl": 0.1, "auc": 0.5048},
    # {"z": 128, "kl": 0.01, "auc": 0.5106},
    # {"z": 128, "kl": 0.001, "auc": 0.5069},
    # {"z": 128, "kl": 0.0001, "auc": 0.504},
    # {"z": 128, "kl": 1e-05, "auc": 0.8852},
    # {"z": 128, "kl": 1e-08, "auc": 0.5050},

    # --- z = 256 ---
    {"z": 256, "kl": 100.0, "auc": 0.8747},  # 0.8761 - ACC:0.5001
    {"z": 256, "kl": 1.0, "auc": 0.8872},
    {"z": 256, "kl": 0.1, "auc": 0.9293},
    {"z": 256, "kl": 0.01, "auc": 0.9303},
    {"z": 256, "kl": 0.001, "auc": 0.9458},
    {"z": 256, "kl": 0.0001, "auc": 0.9326},
    {"z": 256, "kl": 1e-05, "auc": 0.8924},  # 0.9511
    {"z": 256, "kl": 1e-08, "auc": 0.8474},

    # --- z = 512 ---
    {"z": 512, "kl": 100.0, "auc": 0.8347},  # ACC: 0.5000
    {"z": 512, "kl": 1.0, "auc": 0.9182}, # 0.9443 - ACC:0.6936
    {"z": 512, "kl": 0.1, "auc": 0.9452},  # 0.9039
    {"z": 512, "kl": 0.01, "auc": 0.9474}, # 0.9132
    {"z": 512, "kl": 0.001, "auc": 0.9463},
    {"z": 512, "kl": 0.0001, "auc": 0.9442},
    {"z": 512, "kl": 1e-05, "auc": 0.9455}, # # 0.9491 - ACC:0.5735
    {"z": 512, "kl": 1e-08, "auc": 0.8654},

    # # --- z = 1024 (Add your new points here) ---
    {"z": 1024, "kl": 100.0, "auc": 0.8195},
    {"z": 1024, "kl": 1.0, "auc": 0.8898},  # ACC:0.6418
    {"z": 1024, "kl": 0.1, "auc": 0.9147},  # ACC:0.509, Loss: 0.8966
    {"z": 1024, "kl": 0.01, "auc": 0.9311},  # 0.8698, 0.8687  # Finding
    {"z": 1024, "kl": 0.001, "auc": 0.9421},
    {"z": 1024, "kl": 0.0001, "auc": 0.9466},  # 0.9492
    {"z": 1024, "kl": 1e-05, "auc": 0.9445},
    {"z": 1024, "kl": 1e-08, "auc": 0.8724},
]

manual_data_loss = [
    # --- z = 256 ---
    {"z": 256, "kl": 100.0, "auc": 0.8747},
    {"z": 256, "kl": 1.0, "auc": 0.8679},
    {"z": 256, "kl": 0.1, "auc": 0.8819},
    {"z": 256, "kl": 0.01, "auc": 0.8921},
    {"z": 256, "kl": 0.001, "auc": 0.8377},
    {"z": 256, "kl": 0.0001, "auc": 0.9173},
    {"z": 256, "kl": 1e-05, "auc": 0.8924},
    {"z": 256, "kl": 1e-08, "auc": 0.5049},

    # --- z = 512 ---
    {"z": 512, "kl": 100.0, "auc": 0.7528},  # ACC: 0.5085
    {"z": 512, "kl": 1.0, "auc": 0.9182},
    {"z": 512, "kl": 0.1, "auc": 0.9266},  # 0.8685
    {"z": 512, "kl": 0.01, "auc": 0.8976},
    {"z": 512, "kl": 0.001, "auc": 0.9379},  # 0.9052
    {"z": 512, "kl": 0.0001, "auc": 0.9316},
    {"z": 512, "kl": 1e-05, "auc": 0.9455},
    {"z": 512, "kl": 1e-08, "auc": 0.8246},

    # # --- z = 1024 (Add your new points here) ---
    {"z": 1024, "kl": 100.0, "auc": 0.7564},
    {"z": 1024, "kl": 1.0, "auc": 0.8759},  # ACC: 0.5
    {"z": 1024, "kl": 0.1, "auc": 0.8966},
    {"z": 1024, "kl": 0.01, "auc": 0.9169}, # 0.8849
    {"z": 1024, "kl": 0.001, "auc": 0.9410},
    {"z": 1024, "kl": 0.0001, "auc": 0.9466},
    {"z": 1024, "kl": 1e-05, "auc": 0.9347},
    {"z": 1024, "kl": 1e-08, "auc": 0.8465},
]


# ===================== 2) Process Data =====================
df = pd.DataFrame(manual_data)

if df.empty:
    raise ValueError("No data found. Please add points to 'manual_data'.")

# Sort values to ensure the lines connect correctly
df = df.sort_values(["z", "kl"]).reset_index(drop=True)


# ===================== 3) Plotting =====================
# Apply Professional Style Settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 15,
    'axes.titlesize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 12,
    'grid.alpha': 0.5,
    'lines.linewidth': 2.5,  # Slightly thicker lines for visibility
    'lines.markersize': 8
})

# Define custom colors
# color_map = {
#     256:  '#1f77b4',   # Muted Blue (Standard, professional baseline)
#     512:  '#d62728',   # Muted Red (High contrast against blue)
#     1024: '#2ca02c'    # Muted Green (Classic third option)
# }

color_map = {
    256:  '#002244',   # Deep Navy (Very formal)
    512:  '#800000',   # Maroon / Dark Red (High contrast, serious)
    1024: '#004d00'    # Forest Green (Dark organic)
}

# color_map = {
#     256:  '#0055AA',   # Sapphire Blue
#     512:  '#CC3300',   # Burnt Orange / Rust Red
#     1024: '#008844'    # Emerald / Rich Green
# }

# --- Markers ---
marker_map = {
    256:  'o',   # Circle
    512:  's',   # Square
    1024: '^'    # Triangle Up
}

plt.figure(figsize=(8, 6))

# Loop through each unique 'z' dimension to create a separate line
for z_val in sorted(df["z"].unique()):
    # Filter data for this specific z dimension
    g = df[df["z"] == z_val]

    c = color_map.get(z_val, 'black')
    m = marker_map.get(z_val, 'o')

    # Plot x=kl, y=auc
    plt.plot(g["kl"], g["auc"], marker=m, linestyle='-.', color=c, label=f"z={z_val}")

# plt.xlabel(r"KL Weight ($\beta$)", fontsize=15)
# plt.ylabel("AUC", fontsize=15)
# plt.title(r"AUC vs $\beta$", fontsize=18)
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
#
# # Use log scale for X-axis since KL values vary by orders of magnitude
# plt.xscale("log")

plt.legend(title="Latent Dim (z)")
plt.tight_layout()

plt.xlabel(r"KL Weight ($\beta$)", fontsize=21)
plt.ylabel("AUC", fontsize=21)
plt.title(r"AUC vs $\beta$", fontsize=21)
plt.grid(True, which="both", linestyle="--", linewidth=0.6)

# Use log scale for X-axis since KL values vary by orders of magnitude
plt.xscale("log")
plt.legend(title='Latent Dim (z)', loc='lower center', fontsize=18)
plt.tight_layout()

# ===================== 4) Save Figure =====================

save_dir = "/home/chunsup2/PycharmProjects/VIBIO/figure/"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "auc_vs_kl_summary_plot.png")

os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"\nFigure saved to: {save_path}")

plt.show()