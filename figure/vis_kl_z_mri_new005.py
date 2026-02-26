import os
import re
import pandas as pd
import matplotlib.pyplot as plt


raw = r"""


VIB-IO-lr0.005-depth3-kl100.0-z8-ioloss1.0,0.625017,0.0056685177388672
VIB-IO-lr0.005-depth3-kl1.0-z8-ioloss1.0,0.767595,0.0045300630160063
VIB-IO-lr0.005-depth3-kl0.01-z8-ioloss1.0,0.847764,0.0037190948495418
VIB-IO-lr0.005-depth3-kl0.001-z8-ioloss1.0,0.864802,0.0034638260207118
VIB-IO-lr0.005-depth3-kl0.0001-z8-ioloss1.0,0.858624,0.0035511205343956
VIB-IO-lr0.005-depth3-kl1e-08-z8-ioloss1.0,0.798886,0.0043872957655553



VIB-IO-lr0.005-depth3-kl100.0-z16-ioloss1.0,0.576419,0.0057599805646301
VIB-IO-lr0.005-depth3-kl1.0-z16-ioloss1.0,0.806534,0.0041269214174411
VIB-IO-lr0.005-depth3-kl0.01-z16-ioloss1.0,0.850229,0.0037891141433407
VIB-IO-lr0.005-depth3-kl0.001-z16-ioloss1.0,0.872553,0.0033067262183020
VIB-IO-lr0.005-depth3-kl0.0001-z16-ioloss1.0,0.868266,0.0034377119352024
VIB-IO-lr0.005-depth3-kl1e-08-z16-ioloss1.0,0.801644,0.0043177017854150

VIB-IO-lr0.005-depth3-kl100.0-z64-ioloss1.0,0.736988,0.0048258489796245
VIB-IO-lr0.005-depth3-kl1.0-z64-ioloss1.0,0.783030,0.0046249281818670
VIB-IO-lr0.005-depth3-kl0.01-z64-ioloss1.0,0.850957,0.0037171558966807
VIB-IO-lr0.005-depth3-kl0.001-z64-ioloss1.0,0.855961,0.0035670085361456
VIB-IO-lr0.005-depth3-kl0.0001-z64-ioloss1.0,0.795435,0.0043834144751086
VIB-IO-lr0.005-depth3-kl1e-08-z64-ioloss1.0,0.792448,0.0044117328585268

VIB-IO-lr0.005-depth3-kl100.0-z128-ioloss1.0,0.731156,0.0048314197823062
VIB-IO-lr0.005-depth3-kl1.0-z128-ioloss1.0,0.775417,0.0043950942998440
VIB-IO-lr0.005-depth3-kl0.01-z128-ioloss1.0,0.814318,0.0044572862288360
VIB-IO-lr0.005-depth3-kl0.001-z128-ioloss1.0,0.860325,0.0034858743215604
VIB-IO-lr0.005-depth3-kl0.0001-z128-ioloss1.0,0.856098,0.0036012445705153
VIB-IO-lr0.005-depth3-kl1e-08-z128-ioloss1.0,0.817524,0.0041684148209242


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
        "z": int(mz.group(1)),
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
plt.title(r"AUC vs $\beta$ (Data Proportion: 0.005)", fontsize=21)
plt.grid(True, which="both", linestyle="--", linewidth=0.6)

plt.xscale("log")
plt.legend(title='Latent Dim (z)', loc='lower left', fontsize=18)
plt.tight_layout()

# ===================== 4) 保存到指定位置 =====================
# save_path = "/home/zexinji/Zexin/Code/VIB/vib-main/resultsCVS/auc_vs_kl_all_z005_sks.png"  # <<< 改成你的目标路径
save_dir = "/home/chunsup2/PycharmProjects/VIBIO/figure/"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "auc_vs_kl_all_p005_ske.png")

plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {save_path}")

plt.show()