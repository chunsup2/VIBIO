import os
import re
import pandas as pd
import matplotlib.pyplot as plt
# 0.5
raw = r"""

VIB-IO-lr0.0005-depth4-kl100.0-z8-ioloss1.0,0.506537,0.0047010391851769
VIB-IO-lr0.0005-depth4-kl1.0-z8-ioloss1.0,0.504459,0.0033450157841817
VIB-IO-lr0.0005-depth4-kl0.01-z8-ioloss1.0,0.511241,0.0055034358555878
VIB-IO-lr0.0005-depth4-kl0.001-z8-ioloss1.0,0.506205,0.0044367646402809
VIB-IO-lr0.0005-depth4-kl0.0001-z8-ioloss1.0,0.505871,0.0042213555654207
VIB-IO-lr0.0005-depth4-kl1e-05-z8-ioloss1.0,0.50494,0.0037133523432145
VIB-IO-lr0.0005-depth4-kl1e-08-z8-ioloss1.0,0.507365,0.0048085435339103

VIB-IO-lr0.005-depth4-kl100.0-z16-ioloss1.0,0.50848,0.0051555580609173
VIB-IO-lr0.005-depth4-kl1.0-z16-ioloss1.0,0.504885,0.0038290615713602
VIB-IO-lr0.005-depth4-kl0.01-z16-ioloss1.0,0.504674,0.0034111101479452
VIB-IO-lr0.005-depth4-kl0.001-z16-ioloss1.0,0.510929,0.0055540856487192
VIB-IO-lr0.005-depth4-kl0.0001-z16-ioloss1.0,0.518334,0.0055145798421524
VIB-IO-lr0.005-depth4-kl1e-05-z16-ioloss1.0,0.508281,0.0053076683082721
VIB-IO-lr0.005-depth4-kl1e-08-z16-ioloss1.0,0.51311,0.0057935644946508

VIB-IO-lr5e-05-depth4-kl100.0-z64-ioloss1.0,0.504749,0.0035316654894768
VIB-IO-lr5e-05-depth4-kl1.0-z64-ioloss1.0,0.508179,0.0050594640100159
VIB-IO-lr5e-05-depth4-kl0.01-z64-ioloss1.0,0.506493,0.0045858416973546
VIB-IO-lr5e-05-depth4-kl0.001-z64-ioloss1.0,0.512225,0.0058113603551467
VIB-IO-lr5e-05-depth4-kl0.0001-z64-ioloss1.0,0.510119,0.0054029561194138
VIB-IO-lr5e-05-depth4-kl1e-05-z64-ioloss1.0,0.518406,0.0059507471822754
VIB-IO-lr5e-05-depth4-kl1e-08-z64-ioloss1.0,0.50687,0.0047373870073668

VIB-IO-lr0.005-depth4-kl100.0-z128-ioloss1.0,0.531027,0.0056471360552649
VIB-IO-lr0.005-depth4-kl1.0-z128-ioloss1.0,0.522688,0.0057835700669404
VIB-IO-lr0.005-depth4-kl0.01-z128-ioloss1.0,0.875703,0.0035524840271109
VIB-IO-lr0.005-depth4-kl0.001-z128-ioloss1.0,0.520676,0.0057369127915294
VIB-IO-lr0.005-depth4-kl0.0001-z128-ioloss1.0,0.860297,0.0038796502484411
VIB-IO-lr0.005-depth4-kl1e-05-z128-ioloss1.0,0.858287,0.0038903571098898
VIB-IO-lr0.005-depth4-kl1e-08-z128-ioloss1.0,0.837035,0.004066159909234

"""


# ===================== 2) 解析 =====================
num_pat = r"[0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?"
kl_pat = re.compile(rf"kl({num_pat})(?=-z)", re.IGNORECASE)
z_pat  = re.compile(rf"-z({num_pat})(?=-)", re.IGNORECASE)

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
    mz  = z_pat.search(prefix)
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
    raise ValueError("没有解析到任何数据：请检查 raw 内容格式是否正确。")

df["std_5"] = df["std"].map(lambda x: f"{x:.5f}")
df = df.sort_values(["z", "kl"]).reset_index(drop=True)

# 可选：保存解析后的表
df[["z", "kl", "auc", "std_5"]].to_csv("kl_auc_std.csv", index=False, encoding="utf-8-sig")

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


for z_val in sorted(df["z"].unique()):
    g = df[df["z"] == z_val].sort_values("kl")

    c = color_map.get(z_val, 'black')
    m = marker_map.get(z_val, 'o')

    # Plot x=kl, y=auc
    plt.plot(g["kl"], g["auc"], marker=m, linestyle='-.', color=c, label=f"z={z_val}")

    # plt.plot(g["kl"], g["auc"], marker="o", label=f"z={z_val:g}")

plt.xlabel(r"KL Weight ($\beta$)", fontsize=15)
plt.ylabel("AUC", fontsize=15)
plt.title(r"AUC vs $\beta$", fontsize=18)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.xscale("log")
plt.legend(title="Latent Dim (z)")
plt.tight_layout()

# ===================== 4) Save and Plot =====================
# save_path = "/home/zexinji/Zexin/Code/VIB/vib-main/resultsCVS/auc_vs_kl_all_z05_sks.png"  # <<< 改成你的目标路径

save_dir = "/home/chunsup2/PycharmProjects/VIBIO/figure/"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "auc_vs_kl_all_p05_sks.png")

plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {save_path}")

plt.show()