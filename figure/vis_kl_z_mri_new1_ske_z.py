import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ===================== 1) Data Definition =====================
raw = r"""
VIB-IO-lr0.005-depth3-kl100.0-z8-ioloss1.0,0.818121,0.0040679930429309
VIB-IO-lr0.005-depth3-kl1.0-z8-ioloss1.0,0.908199,0.002769003611441
VIB-IO-lr0.005-depth3-kl0.01-z8-ioloss1.0,0.915045,0.0026441103530238
VIB-IO-lr0.005-depth3-kl0.001-z8-ioloss1.0,0.91705,0.0026182189476648
VIB-IO-lr0.005-depth3-kl0.0001-z8-ioloss1.0,0.912853,0.002681191524755
VIB-IO-lr0.005-depth3-kl1e-05-z8-ioloss1.0,0.914165,0.0026695579566939
VIB-IO-lr0.005-depth3-kl1e-08-z8-ioloss1.0,0.90508,0.0028457651392163

VIB-IO-lr0.005-depth3-kl100.0-z16-ioloss1.0,0.796904,0.0037155112257531
VIB-IO-lr0.005-depth3-kl1.0-z16-ioloss1.0,0.863199,0.0035062795569922
VIB-IO-lr0.005-depth3-kl0.01-z16-ioloss1.0,0.914803,0.0026611466052379
VIB-IO-lr0.005-depth3-kl0.001-z16-ioloss1.0,0.913389,0.0026618614912181
VIB-IO-lr0.005-depth3-kl0.0001-z16-ioloss1.0,0.912382,0.0027004995818635
VIB-IO-lr0.005-depth3-kl1e-05-z16-ioloss1.0,0.913081,0.0026721885664669
VIB-IO-lr0.005-depth3-kl1e-08-z16-ioloss1.0,0.898631,0.0029614098475651

VIB-IO-lr0.005-depth3-kl100.0-z64-ioloss1.0,0.813748,0.0041542612653533
VIB-IO-lr0.005-depth3-kl1.0-z64-ioloss1.0,0.869254,0.0034434900311244
VIB-IO-lr0.005-depth3-kl0.01-z64-ioloss1.0,0.909462,0.0027393807629317
VIB-IO-lr0.005-depth3-kl0.001-z64-ioloss1.0,0.911781,0.0027120825649414
VIB-IO-lr0.005-depth3-kl0.0001-z64-ioloss1.0,0.907563,0.0027815227988323
VIB-IO-lr0.005-depth3-kl1e-05-z64-ioloss1.0,0.910835,0.0026901337371412
VIB-IO-lr0.005-depth3-kl1e-08-z64-ioloss1.0,0.895265,0.002987111604981581

VIB-IO-lr0.005-depth3-kl100.0-z128-ioloss1.0,0.895121,0.0030565669854497
VIB-IO-lr0.005-depth3-kl1.0-z128-ioloss1.0,0.906249,0.0028110569512018
VIB-IO-lr0.005-depth3-kl0.01-z128-ioloss1.0,0.907063,0.0027591553300471
VIB-IO-lr0.005-depth3-kl0.001-z128-ioloss1.0,0.908147,0.0027605840190139
VIB-IO-lr0.005-depth3-kl0.0001-z128-ioloss1.0,0.907171,0.0028017054985191
VIB-IO-lr0.005-depth3-kl1e-05-z128-ioloss1.0,0.905759,0.0028003543426167
VIB-IO-lr0.005-depth3-kl1e-08-z128-ioloss1.0,0.888242,0.003090383916044
"""

# ===================== 2) Parse Data =====================
num_pat = r"[0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?"
kl_pat = re.compile(rf"kl({num_pat})(?=-z)", re.IGNORECASE)
z_pat = re.compile(rf"-z({num_pat})(?=-)", re.IGNORECASE)

rows = []
for line in raw.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue

    try:
        prefix, auc_s, std_s = line.rsplit(",", 2)
    except ValueError:
        continue

    mkl = kl_pat.search(prefix)
    mz = z_pat.search(prefix)
    if not (mkl and mz):
        continue

    rows.append({
        "z": float(mz.group(1)),
        "kl": float(mkl.group(1)),
        "auc": float(auc_s),
        "std": float(std_s),
        "tag": prefix
    })

df = pd.DataFrame(rows)
if df.empty:
    raise ValueError("No data parsed.")

df = df.sort_values(["kl", "z"]).reset_index(drop=True)

# ===================== 3) Plotting =====================
# Apply Professional Style Settings
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
    'lines.markersize': 8
})

plt.figure(figsize=(9, 7))

# Get unique Beta values
unique_kls = sorted(df["kl"].unique(), reverse=True)

# === Distinct Colors (High Contrast) ===
# These are manually selected to be maximally distinct
distinct_colors = [
    '#E6194B',  # Red
    '#3CB44B',  # Green
    '#FFE119',  # Yellow
    '#4363D8',  # Blue
    '#F58231',  # Orange
    '#911EB4',  # Purple
    '#42D4F4',  # Cyan
    '#F032E6',  # Magenta
    '#000000'  # Black
]

# === Distinct Markers ===
distinct_markers = ['o', 's', '^', 'D', 'v', 'X', '*']

for i, kl_val in enumerate(unique_kls):
    g = df[df["kl"] == kl_val].sort_values("z")

    # Cycle through distinct colors and markers
    c = distinct_colors[i % len(distinct_colors)]
    m = distinct_markers[i % len(distinct_markers)]

    # Label formatting
    if kl_val == 100.0:
        label_str = r"$\beta$=100"
    elif kl_val == 1.0:
        label_str = r"$\beta$=1.0"
    elif kl_val < 0.0001:
        label_str = f"$\\beta$={kl_val:.0e}"  # Scientific notation for tiny nums
    else:
        label_str = f"$\\beta$={kl_val}"

    plt.plot(g["z"], g["auc"], marker=m, linestyle='--', color=c, label=label_str, linewidth=2)

# Labels
plt.xlabel(r"Latent Dimension ($z$)", fontsize=21)
plt.ylabel("AUC", fontsize=21)
plt.title(r"AUC vs $z$ (Fixed $\beta$)", fontsize=21)
plt.grid(True, which="both", linestyle="--", linewidth=0.6)

# X-Axis Scaling
plt.xscale("log", base=2)
unique_zs = sorted(df["z"].unique())
plt.xticks(unique_zs, [str(int(z)) for z in unique_zs])  # Force integer labels
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())

# Legend
plt.legend(title=r'KL Weight ($\beta$)', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.tight_layout()

# ===================== 4) Save =====================
save_dir = "/home/chunsup2/PycharmProjects/VIBIO/figure/"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "auc_vs_z_fixed_beta_colors.png")

plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {save_path}")

plt.show()