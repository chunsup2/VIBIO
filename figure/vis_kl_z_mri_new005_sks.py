import os
import re
import pandas as pd
import matplotlib.pyplot as plt


raw = r"""



VIB-IO-lr0.0005-depth4-kl100.0-z8-ioloss1.0,0.512953,0.0055414038861638
VIB-IO-lr0.0005-depth4-kl1.0-z8-ioloss1.0,0.506655,0.0045721834479826
VIB-IO-lr0.0005-depth4-kl0.01-z8-ioloss1.0,0.509102,0.0052560835091023
VIB-IO-lr0.0005-depth4-kl0.001-z8-ioloss1.0,0.508354,0.0052638283909599
VIB-IO-lr0.0005-depth4-kl0.0001-z8-ioloss1.0,0.508396,0.0052020625867427
VIB-IO-lr0.0005-depth4-kl1e-05-z8-ioloss1.0,0.510911,0.005394102672299
VIB-IO-lr0.0005-depth4-kl1e-08-z8-ioloss1.0,0.511575,0.005691866322701



VIB-IO-lr0.005-depth4-kl100.0-z16-ioloss1.0,0.505623,0.0040168429977457
VIB-IO-lr0.005-depth4-kl1.0-z16-ioloss1.0,0.504794,0.0035357922635284
VIB-IO-lr0.005-depth4-kl0.01-z16-ioloss1.0,0.515492,0.0055489969684546
VIB-IO-lr0.005-depth4-kl0.001-z16-ioloss1.0,0.740951,0.0049441103531691
VIB-IO-lr0.005-depth4-kl0.0001-z16-ioloss1.0,0.510259,0.00556464648644
VIB-IO-lr0.005-depth4-kl1e-05-z16-ioloss1.0,0.507916,0.0051461653268452
VIB-IO-lr0.005-depth4-kl1e-08-z16-ioloss1.0,0.752788,0.004902538477883


VIB-IO-lr0.005-depth4-kl100.0-z64-ioloss1.0,0.514846,0.0055897997652689
VIB-IO-lr0.005-depth4-kl1.0-z64-ioloss1.0,0.521142,0.0057038550001019
VIB-IO-lr0.005-depth4-kl0.01-z64-ioloss1.0,0.541212,0.0055451979688103
VIB-IO-lr0.005-depth4-kl0.001-z64-ioloss1.0,0.530983,0.0055052791501096
VIB-IO-lr0.005-depth4-kl0.0001-z64-ioloss1.0,0.504649,0.0035047311283647
VIB-IO-lr0.005-depth4-kl1e-05-z64-ioloss1.0,0.505965,0.0042119178569835
VIB-IO-lr0.005-depth4-kl1e-08-z64-ioloss1.0,0.504942,0.0037260041236301254


VIB-IO-lr0.005-depth4-kl100.0-z128-ioloss1.0,0.51397,0.0055366612423767
VIB-IO-lr0.005-depth4-kl1.0-z128-ioloss1.0,0.50807,0.0049804235516951
VIB-IO-lr0.005-depth4-kl0.01-z128-ioloss1.0,0.513081,0.0055414726581313
VIB-IO-lr0.005-depth4-kl0.001-z128-ioloss1.0,0.513508,0.0056109444082701
VIB-IO-lr0.005-depth4-kl0.0001-z128-ioloss1.0,0.50718,0.0046232822975007
VIB-IO-lr0.005-depth4-kl1e-05-z128-ioloss1.0,0.513625,0.0055932029540262
VIB-IO-lr0.005-depth4-kl1e-08-z128-ioloss1.0,0.504616,0.0034947566665279


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
plt.figure()

for z_val in sorted(df["z"].unique()):
    g = df[df["z"] == z_val].sort_values("kl")
    plt.plot(g["kl"], g["auc"], marker="o", label=f"z={z_val:g}")

plt.xlabel("β")
plt.ylabel("AUC")
plt.title("AUC vs β")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xscale("log")
plt.legend(title="z", loc="upper right")
plt.tight_layout()

# ===================== 4) 保存到指定位置 =====================
# save_path = "/home/zexinji/Zexin/Code/VIB/vib-main/resultsCVS/auc_vs_kl_all_z005_sks.png"  # <<< 改成你的目标路径
save_dir = "/home/chunsup2/PycharmProjects/VIBIO/figure/"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "auc_vs_kl_all_p005_sks.png")

plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {save_path}")

plt.show()