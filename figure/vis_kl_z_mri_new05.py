import os
import re
import pandas as pd
import matplotlib.pyplot as plt
# 0.5
# VIB-IO-lr0.005-depth3-kl0.01-z8-ioloss1.0,0.913008,0.0027142437061321

# SKE MRI VIB-CE choose:
# dataproprotion 1:VIB-CE-lr5e-05-depth3-z16-kl0.1,0.917011,0.0025965125190212
# dataproprotion 0.5:VIB-CE-lr5e-05-depth3-z16-kl0.05,0.914329,0.0026527388551223
# dataproprotion 0.005:VIB-CE-lr0.001-depth3-z8-kl0.0005,0.87018,0.0034096089182702

raw = r"""


VIB-IO-lr0.005-depth3-kl100.0-z8-ioloss1.0,0.783463,0.0043589895892767
VIB-IO-lr0.005-depth3-kl1.0-z8-ioloss1.0,0.872603,0.0033608827673234

VIB-IO-lr0.005-depth3-kl0.01-z8-ioloss1.0,0.91440,0.0026577216506287
VIB-IO-lr0.005-depth3-kl0.001-z8-ioloss1.0,0.910425,0.0027548715471397
VIB-IO-lr0.005-depth3-kl0.0001-z8-ioloss1.0,0.9112,0.0027102359740666
VIB-IO-lr0.005-depth3-kl1e-05-z8-ioloss1.0,0.910212,0.0027200009393296
VIB-IO-lr0.005-depth3-kl1e-08-z8-ioloss1.0,0.897653,0.0029908867920104




VIB-IO-lr0.005-depth3-kl100.0-z16-ioloss1.0,0.792752,0.0042935708715242
VIB-IO-lr0.005-depth3-kl1.0-z16-ioloss1.0,0.8677,0.0034431732572561
VIB-IO-lr0.005-depth3-kl0.01-z16-ioloss1.0,0.912511,0.0026778651212157
VIB-IO-lr0.005-depth3-kl0.001-z16-ioloss1.0,0.91273,0.0026924771391893
VIB-IO-lr0.005-depth3-kl0.0001-z16-ioloss1.0,0.91016,0.0027254275139757
VIB-IO-lr0.005-depth3-kl1e-05-z16-ioloss1.0,0.910371,0.0027163476823022
VIB-IO-lr0.005-depth3-kl1e-08-z16-ioloss1.0,0.894305,0.0030354575671938


VIB-IO-lr0.005-depth3-kl100.0-z64-ioloss1.0,0.850672,0.0037399004101362
VIB-IO-lr0.005-depth3-kl1.0-z64-ioloss1.0,0.891623,0.003098022024844848
VIB-IO-lr0.005-depth3-kl0.01-z64-ioloss1.0,0.907259,0.0027756101970933
VIB-IO-lr0.005-depth3-kl0.001-z64-ioloss1.0,0.90929,0.0027546552226526
VIB-IO-lr0.005-depth3-kl0.0001-z64-ioloss1.0,0.906041,0.0028490319484989
VIB-IO-lr0.005-depth3-kl1e-05-z64-ioloss1.0,0.906006,0.00284551355273
VIB-IO-lr0.005-depth3-kl1e-08-z64-ioloss1.0,0.890625,0.0031035527589634


VIB-IO-lr0.005-depth3-kl100.0-z128-ioloss1.0,0.88261,0.0032611437245972
VIB-IO-lr0.005-depth3-kl1.0-z128-ioloss1.0,0.899726,0.0029554548252896
VIB-IO-lr0.005-depth3-kl0.01-z128-ioloss1.0,0.898947,0.0029130026348922
VIB-IO-lr0.005-depth3-kl0.001-z128-ioloss1.0,0.905494,0.002817528088318
VIB-IO-lr0.005-depth3-kl0.0001-z128-ioloss1.0,0.905706,0.0028369827146757
VIB-IO-lr0.005-depth3-kl1e-05-z128-ioloss1.0,0.903835,0.0028435322400428
VIB-IO-lr0.005-depth3-kl1e-08-z128-ioloss1.0,0.888019,0.0031253791713965




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
plt.title(r"AUC vs $\beta$ (Data Proportion: 0.5)", fontsize=21)
plt.grid(True, which="both", linestyle="--", linewidth=0.6)

plt.xscale("log")
plt.legend(title='Latent Dim (z)', loc='lower left', fontsize=18)
plt.tight_layout()

# ===================== 4) Save path =====================
save_dir = "/home/chunsup2/PycharmProjects/VIBIO/figure/"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "auc_vs_kl_all_p05_ske.png")

plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {save_path}")

plt.show()