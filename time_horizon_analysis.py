import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy import stats

horizons = {"2min": 120, "3.5min" : 210, "5min": 300, "7.5min" : 450, "10min" : 600}

# folder_name = "jpm_data_2022_january"
folder_name = "/scratch0/hanjames"

message_files = []
order_files = []

for file in sorted(os.listdir(folder_name)):
    if file.endswith("_message_10.csv") and file.startswith("JPM_2025"):
        message_files.append(file)
    elif file.endswith("_orderbook_10.csv") and file.startswith("JPM_2025"):
        order_files.append(file)

print(f"Total files loaded in message file: {len(message_files)}")
print(f"Total files loaded in order file: {len(order_files)}")

all_jump_flags = []
all_mid_prices = []
all_log_returns = []
all_timestamps = []
all_mid_prices_nonzero = []
z_values = []
delta_values = []

jpm_returns_with_time = {}

for msg_file, ord_file in zip(message_files, order_files):
    jpm_message_file = pd.read_csv(os.path.join(folder_name, msg_file), header=None, low_memory=False)
    jpm_order_file = pd.read_csv(os.path.join(folder_name, ord_file), header=None, low_memory=False)
    jpm_order_file.replace([9999999999, -9999999999, 0], float('nan'), inplace=True)
    message_file_timestamps = jpm_message_file[0]
    mask = (message_file_timestamps >= 34800) & (message_file_timestamps <= 57000)
    jpm_message_file = jpm_message_file[mask].reset_index(drop=True)
    jpm_order_file = jpm_order_file[mask].reset_index(drop=True)
    jpm_mid_price = (jpm_order_file[0] + jpm_order_file[2]) / 2 / 10000
    jpm_midprice_with_time = pd.DataFrame({
        "timestamp": jpm_message_file[0].values,
        "mid_price": jpm_mid_price.values
    })
    jpm_midprice_with_time = jpm_midprice_with_time.set_index(pd.to_datetime(jpm_midprice_with_time["timestamp"], unit='s'))

    for label, seconds in horizons.items():
        resampled = jpm_midprice_with_time["mid_price"].resample(str(seconds) + 's').last()
        resampled_returns = np.log(resampled / resampled.shift(1)).dropna()
        if seconds not in jpm_returns_with_time:
            jpm_returns_with_time[seconds] = []
        jpm_returns_with_time[seconds].append(resampled_returns)

jpm_returns_combined = {}
for seconds, list_of_series in jpm_returns_with_time.items():
    jpm_returns_combined[seconds] = pd.concat(list_of_series, ignore_index=True)

for seconds, label in zip(horizons.values(), horizons.keys()):
    print(f"{label}: {len(jpm_returns_combined[seconds])} returns, mean={jpm_returns_combined[seconds].mean():.6f}, std={jpm_returns_combined[seconds].std():.6f}")
    returns = jpm_returns_combined[seconds]
    kurt = stats.kurtosis(returns)
    skew = stats.skew(returns)
    print(f"{label}: kurtosis={kurt:.3f}, skew={skew:.3f}")

fig, axes = plt.subplots(1, len(horizons), figsize=(20, 4), sharey=False)

for ax, (label, seconds) in zip(axes, horizons.items()):
    returns = jpm_returns_combined[seconds]
    ax.hist(returns, bins=60, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_yscale('log')
    ax.set_title(f"{label} returns\nkurtosis={stats.kurtosis(returns):.2f}")
    ax.set_xlabel("Log return")

axes[0].set_ylabel("Frequency (log scale)")
plt.tight_layout()
plt.savefig("horizon_return_distributions.png", dpi=150)
# plt.show()

for seconds, label in zip(horizons.values(), horizons.keys()):
    returns = jpm_returns_combined[seconds]
    sigma_h = returns.std()
    print(f"\n{label} (sigma={sigma_h:.6f}):")
    for k in [2, 3, 4]:
        threshold = k * sigma_h
        n_flagged = (returns.abs() > threshold).sum()
        pct_flagged = n_flagged / len(returns) * 100
        print(f"  k={k}: {n_flagged} flagged ({pct_flagged:.2f}%)")