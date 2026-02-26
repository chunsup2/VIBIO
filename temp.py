import matplotlib.pyplot as plt
import pandas as pd
import re

# Your data as a string
raw_data = """
VIB-IO-lr0.0005-depth4-kl100.0-z168-ioloss1.0,0.50883,0.0051976254107962
VIB-IO-lr0.0005-depth4-kl10.0-z168-ioloss1.0,0.504838,0.0037484066137109
VIB-IO-lr0.0005-depth4-kl1.0-z168-ioloss1.0,0.504565,0.0034196611892541
VIB-IO-lr0.0005-depth4-kl0.1-z168-ioloss1.0,0.513061,0.0056904888762218
VIB-IO-lr0.0005-depth4-kl0.01-z168-ioloss1.0,0.505607,0.0040850666227092
VIB-IO-lr0.0005-depth4-kl0.001-z168-ioloss1.0,0.504725,0.0034703125072877
VIB-IO-lr0.0005-depth4-kl0.0001-z168-ioloss1.0,0.505177,0.0039042594135038
VIB-IO-lr0.0005-depth4-kl1e-08-z168-ioloss1.0,0.535301,0.0055871640555699
VIB-IO-lr0.005-depth4-kl100.0-z168-ioloss1.0,0.51664,0.0057686985948182
VIB-IO-lr0.005-depth4-kl10.0-z168-ioloss1.0,0.507863,0.0047573847713405
VIB-IO-lr0.005-depth4-kl1.0-z168-ioloss1.0,0.510887,0.0056262676502038
VIB-IO-lr0.005-depth4-kl0.1-z168-ioloss1.0,0.645206,0.0055564526606419
VIB-IO-lr0.005-depth4-kl0.01-z168-ioloss1.0,0.508351,0.0050700124079966
VIB-IO-lr0.005-depth4-kl0.001-z168-ioloss1.0,0.505081,0.0038877286955079
VIB-IO-lr0.005-depth4-kl0.0001-z168-ioloss1.0,0.855624,0.0039649008423966
VIB-IO-lr0.005-depth4-kl1e-05-z168-ioloss1.0,0.514281,0.0059093120769994
VIB-IO-lr0.005-depth4-kl1e-08-z168-ioloss1.0,0.553465,0.00572142615454
VIB-IO-lr0.005-depth4-kl10.0-z64-ioloss1.0,0.506495,0.0046186885620715
VIB-IO-lr0.005-depth4-kl100.0-z8-ioloss1.0,0.504719,0.0035636963252591
VIB-IO-lr0.005-depth4-kl10.0-z8-ioloss1.0,0.504871,0.0036529094253245
VIB-IO-lr0.005-depth4-kl1.0-z8-ioloss1.0,0.507587,0.004901018925759
VIB-IO-lr0.005-depth4-kl0.1-z8-ioloss1.0,0.504922,0.0036908419568325
VIB-IO-lr0.005-depth4-kl0.01-z8-ioloss1.0,0.504734,0.0036266235503566
VIB-IO-lr0.005-depth4-kl0.001-z8-ioloss1.0,0.510179,0.0054334418187259
VIB-IO-lr0.005-depth4-kl0.0001-z8-ioloss1.0,0.514969,0.0057385077429053
VIB-IO-lr0.005-depth4-kl1e-05-z8-ioloss1.0,0.516513,0.0057573368523948
VIB-IO-lr0.005-depth4-kl1e-08-z8-ioloss1.0,0.504735,0.0036415448057485
"""

# # Parse the data
# parsed_data = []
# lines = raw_data.strip().split('\n')
#
# for line in lines:
#     parts = line.split(',')
#     config_str = parts[0]
#     mean_val = float(parts[1])
#     std_val = float(parts[2])
#
#     # Extract parameters using regex to handle scientific notation and variable lengths
#     # We look for the value between the key (e.g., 'kl') and the next separator/key (e.g., '-z')
#     lr_val = float(re.search(r'lr(.+?)-depth', config_str).group(1))
#     kl_val = float(re.search(r'kl(.+?)-z', config_str).group(1))
#     z_val = int(re.search(r'z(.+?)-ioloss', config_str).group(1))
#
#     parsed_data.append({
#         'lr': lr_val,
#         'kl': kl_val,
#         'z': z_val,
#         'mean': mean_val,
#         'std': std_val
#     })
#
# # Create DataFrame
# df = pd.DataFrame(parsed_data)
#
# # Sort by KL weight for proper line plotting
# df = df.sort_values(by='kl')
#
# # Plotting
# plt.figure(figsize=(12, 7))
#
# # Define the baseline value
# baseline_value = 0.75
#
# # Group by learning rate (lr) and latent dim (z) to draw separate lines
# groups = df.groupby(['lr', 'z'])
#
# for (lr, z), group_df in groups:
#     label = f'lr={lr}, z={z}'
#     plt.errorbar(group_df['kl'], group_df['mean'], yerr=group_df['std'],
#                  label=label, marker='o', capsize=5, alpha=0.8)
#
# # Add the baseline reference line
# plt.axhline(y=baseline_value, color='red', linestyle='--', linewidth=2, label=f'Baseline {baseline_value}')
#
# # Formatting the plot
# plt.xscale('log')  # Use log scale for KL because values vary from 1e-8 to 100
# plt.xlabel('KL Weight (log scale)', fontsize=12)
# plt.ylabel('Mean Performance', fontsize=12)
# plt.title(f'Performance vs KL Weight (Baseline: {baseline_value})', fontsize=14)
# plt.legend(title='Parameters')
# plt.grid(True, which="both", ls="-", alpha=0.2)
#
# plt.show()


# 1. Prepare the data

parsed = []
for line in raw_data:
    parts = line.split(',')
    parsed.append({'name': parts[0], 'mean': float(parts[1]), 'std': float(parts[2])})

df = pd.DataFrame(parsed)

# 2. Sort results to show the distribution of success vs failure
df = df.sort_values(by='mean', ascending=False).reset_index(drop=True)

# 3. Plotting
plt.figure(figsize=(12, 6))

threshold = 0.75
# Color: Green for success, Red for failure
colors = ['#2ecc71' if m >= threshold else '#e74c3c' for m in df['mean']]

plt.bar(range(len(df)), df['mean'], color=colors, edgecolor='black', alpha=0.8)

# Add reference lines
plt.axhline(y=threshold, color='black', linestyle='--', label=f'Target ({threshold})')
plt.axhline(y=0.5, color='blue', linestyle=':', label='Chance Level (0.5)')

plt.title('Training Stability Analysis: Success Rarity', fontsize=14)
plt.ylabel('Mean Performance')
plt.xlabel('Individual Attempts (Sorted by Result)')
plt.xticks([]) # Hide x-labels as they are too many/long
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
# plt.savefig('stability_plot.png')