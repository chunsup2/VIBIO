import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# SKE MRI VIB-CE choose:
# dataproprotion 1:VIB-CE-lr5e-05-depth3-z16-kl0.1,0.917011,0.0025965125190212
# dataproprotion 0.5:VIB-CE-lr5e-05-depth3-z16-kl0.05,0.914329,0.0026527388551223
# dataproprotion 0.005:VIB-CE-lr0.001-depth3-z8-kl0.0005,0.87018,0.0034096089182702

# 1.0

# VIB-IO-lr0.005-depth3-kl0.001-z8-ioloss1.0,0.915982,0.0026062762826256
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

# ===================== 3) 画图 =====================
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
    8: '#002244',  # Deep Navy (Very formal)
    16: '#800000',  # Maroon / Dark Red (High contrast, serious)
    64: '#004d00',  # Forest Green (Dark organic)
    128: '#8B5A00'    # Deep Purple (New addition)
}

# color_map = {
#     256:  '#0055AA',   # Sapphire Blue
#     512:  '#CC3300',   # Burnt Orange / Rust Red
#     1024: '#008844'    # Emerald / Rich Green
# }

# --- Markers ---
marker_map = {
    8: 'o',  # Circle
    16: 's',  # Square
    64: '^',  # Triangle Up
    128: 'D'    # Diamond (New addition)
}

plt.figure(figsize=(8, 6))

for z_val in sorted(df["z"].unique()):
    g = df[df["z"] == z_val].sort_values("kl")

    c = color_map.get(z_val, 'black')
    m = marker_map.get(z_val, 'o')

    # Plot x=kl, y=auc
    plt.plot(g["kl"], g["auc"], marker=m, linestyle='-.', color=c, label=f"z={z_val}")

    # plt.plot(g["kl"], g["auc"], marker="o", label=f"z={z_val:g}")

plt.xlabel(r"KL Weight ($\beta$)", fontsize=21)
plt.ylabel("AUC", fontsize=21)
plt.title(r"AUC vs $\beta$", fontsize=21)
plt.grid(True, which="both", linestyle="--", linewidth=0.6)

plt.xscale("log")
plt.legend(title='Latent Dim (z)', loc='lower left', fontsize=18)
plt.tight_layout()

# ===================== 4) 保存到指定位置 =====================
# save_path = "/home/zexinji/Zexin/Code/VIB/vib-main/resultsCVS/auc_vs_kl_all_z1_sks.png"  # <<< 改成你的目标路径
save_dir = "/home/chunsup2/PycharmProjects/VIBIO/figure/"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "auc_vs_kl_all_p1_ske.png")

plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {save_path}")

plt.show()