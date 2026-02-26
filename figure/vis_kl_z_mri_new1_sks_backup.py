import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# 1.0
raw = r"""


VIB-IO-lr5e-05-depth4-kl100.0-z8-ioloss1.0,0.504888,0.0036435230109743
VIB-IO-lr5e-05-depth4-kl1.0-z8-ioloss1.0,0.505322,0.0038765515865443
VIB-IO-lr5e-05-depth4-kl0.01-z8-ioloss1.0,0.510594,0.0055086897564238
VIB-IO-lr5e-05-depth4-kl0.001-z8-ioloss1.0,0.506939,0.0046259796503677
VIB-IO-lr5e-05-depth4-kl0.0001-z8-ioloss1.0,0.507642,0.0049271442373548
VIB-IO-lr5e-05-depth4-kl1e-05-z8-ioloss1.0,0.560403,0.0058665344101343
VIB-IO-lr5e-05-depth4-kl1e-08-z8-ioloss1.0,0.505986,0.0041909245430655



VIB-IO-lr5e-05-depth4-kl100.0-z16-ioloss1.0,0.505584,0.0040881389548364
VIB-IO-lr5e-05-depth4-kl1.0-z16-ioloss1.0,0.507097,0.0046880272398051
VIB-IO-lr5e-05-depth4-kl0.01-z16-ioloss1.0,0.85434,0.0038764343755296
VIB-IO-lr5e-05-depth4-kl0.001-z16-ioloss1.0,0.509157,0.0052399611164909
VIB-IO-lr5e-05-depth4-kl0.0001-z16-ioloss1.0,0.504912,0.0036326833905938
VIB-IO-lr5e-05-depth4-kl1e-05-z16-ioloss1.0,0.509964,0.0055415536228249
VIB-IO-lr5e-05-depth4-kl1e-08-z16-ioloss1.0,0.507067,0.0047327090260613


VIB-IO-lr0.0005-depth4-kl100.0-z64-ioloss1.0,0.507897,0.005120944022556
VIB-IO-lr0.0005-depth4-kl1.0-z64-ioloss1.0,0.515811,0.0059163157819753
VIB-IO-lr0.0005-depth4-kl0.01-z64-ioloss1.0,0.881063,0.0035451375914384
VIB-IO-lr0.0005-depth4-kl0.001-z64-ioloss1.0,0.513947,0.0056146836348438
VIB-IO-lr0.0005-depth4-kl0.0001-z64-ioloss1.0,0.858265,0.0038439220976299
VIB-IO-lr0.0005-depth4-kl1e-05-z64-ioloss1.0,0.863801,0.0038380783558328
VIB-IO-lr0.0005-depth4-kl1e-08-z64-ioloss1.0,0.528472,0.0058162163505902



VIB-IO-lr5e-05-depth4-kl100.0-z128-ioloss1.0,0.506294,0.0045949451873398
VIB-IO-lr5e-05-depth4-kl1.0-z128-ioloss1.0,0.867741,0.0037205074152675
VIB-IO-lr5e-05-depth4-kl0.01-z128-ioloss1.0,0.854849,0.0039291929221733
VIB-IO-lr5e-05-depth4-kl0.001-z128-ioloss1.0,0.839172,0.0041126096580518
VIB-IO-lr5e-05-depth4-kl0.0001-z128-ioloss1.0,0.514863,0.0057252745231591
VIB-IO-lr5e-05-depth4-kl1e-05-z128-ioloss1.0,0.847682,0.0039900749089775
VIB-IO-lr5e-05-depth4-kl1e-08-z128-ioloss1.0,0.80781,0.0044428795721178



"""



# ===================== 2) Parsing =====================
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
plt.legend(title="z")
plt.tight_layout()

# ===================== 4) 保存到指定位置 =====================
# save_path = "/home/zexinji/Zexin/Code/VIB/vib-main/resultsCVS/auc_vs_kl_all_z1_sks.png"  # <<< 改成你的目标路径

save_dir = "/home/chunsup2/PycharmProjects/VIBIO/figure/"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "auc_vs_kl_all_p1_sks.png")

plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {save_path}")

plt.show()