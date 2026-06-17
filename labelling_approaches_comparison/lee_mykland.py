import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import math

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

c = math.sqrt(2 / math.pi)
alpha = 0.05

all_jump_flags = []
all_mid_prices = []
all_log_returns = []
all_timestamps = []
all_mid_prices_nonzero = []
z_values = []
delta_values = []

for msg_file, ord_file in zip(message_files, order_files):
    jpm_message_file = pd.read_csv(os.path.join(folder_name, msg_file), header=None, low_memory=False)
    jpm_order_file = pd.read_csv(os.path.join(folder_name, ord_file), header=None, low_memory=False)

    jpm_order_file.replace([9999999999, -9999999999, 0], float('nan'), inplace=True)

    message_file_timestamps = jpm_message_file[0]
    mask = (message_file_timestamps >= 34800) & (message_file_timestamps <= 57000)
    jpm_message_file = jpm_message_file[mask].reset_index(drop=True)
    jpm_order_file = jpm_order_file[mask].reset_index(drop=True)

    jpm_mid_price = (jpm_order_file[0] + jpm_order_file[2]) / 2 / 10000

    jpm_log_returns = np.log(jpm_mid_price / jpm_mid_price.shift(1)).dropna()
    jpm_log_returns = jpm_log_returns[jpm_log_returns != 0].reset_index(drop=True)
    if len(jpm_log_returns) == 0:
        print(f"Skipping {msg_file} because no non-zero returns")
        continue
    all_mid_prices_nonzero.append(jpm_mid_price.iloc[jpm_log_returns.index])

    n_obs = len(jpm_log_returns)
    lm_window_size = math.ceil(math.sqrt(252 * n_obs))

    jpm_abs_returns = np.abs(jpm_log_returns)
    jpm_bipower_variation = (jpm_abs_returns * jpm_abs_returns.shift(1)).rolling(window=lm_window_size).sum() / (lm_window_size - 2)
    jpm_instantaneous_volatility = np.sqrt(jpm_bipower_variation).replace(0, np.nan)

    location_parameter = ((math.sqrt(2 * math.log(n_obs)))/c) - (math.log(math.pi) + math.log(math.log(n_obs))) / (2 * c * math.sqrt(2 * math.log(n_obs)))
    scale_parameter = 1 / (c * math.sqrt(2 * math.log(n_obs)))

    jpm_test_statistic = jpm_log_returns / jpm_instantaneous_volatility
    jpm_standardized_test_statistic = (np.abs(jpm_test_statistic) - location_parameter) / scale_parameter
    jpm_critical_value = -np.log(-np.log(1 - alpha))
    jpm_jump_flag = (jpm_standardized_test_statistic > jpm_critical_value).astype(int)

    jpm_jump_indices = jpm_jump_flag[jpm_jump_flag == 1].index
    jpm_total_ticks_per_day = len(jpm_log_returns)

    min_gap = 10
    filtered_jump_ticks = []
    last_jump = -min_gap

    for tick in jpm_jump_indices:
        if tick - last_jump >= min_gap:
            filtered_jump_ticks.append(tick)
            last_jump = tick

    filtered_jump_ticks = np.array(filtered_jump_ticks)

    for i, tick in enumerate(filtered_jump_ticks):
        if i + 1 < len(filtered_jump_ticks):
            z_values.append(filtered_jump_ticks[i + 1] - tick)
            delta_values.append(1)
        else:
            z_values.append(jpm_total_ticks_per_day - tick)
            delta_values.append(0)

    all_jump_flags.append(jpm_jump_flag)
    all_mid_prices.append(jpm_mid_price)
    all_log_returns.append(jpm_log_returns)
    all_timestamps.append(jpm_message_file[0][jpm_log_returns.index])

all_jump_flags = pd.concat(all_jump_flags, ignore_index=True)
all_mid_prices = pd.concat(all_mid_prices, ignore_index=True)
all_log_returns = pd.concat(all_log_returns, ignore_index=True)
all_timestamps = pd.concat(all_timestamps, ignore_index=True)
all_mid_prices_nonzero = pd.concat(all_mid_prices_nonzero, ignore_index=True)

print(f"\nTotal ticks: {len(all_jump_flags)}")
print(f"Total jumps detected: {all_jump_flags.sum()}")
print(f"Jump rate: {all_jump_flags.mean():.4%}")
print(f"Critical value: {jpm_critical_value:.4f}")
print(f"Window size K (last day): {lm_window_size}")
print(f"Location parameter Cn (last day): {location_parameter:.4f}")
print(f"Scale parameter Sn (last day): {scale_parameter:.4f}")

plt.figure(figsize=(15, 5))
plt.plot(all_mid_prices_nonzero.values, linewidth=0.5, color='blue', label='Mid Price')
jump_locations = all_jump_flags[all_jump_flags == 1].index
plt.scatter(jump_locations, all_mid_prices_nonzero.iloc[jump_locations].values, color='red', s=30, zorder=5, label='Jumps', marker='x')
plt.title('Mid Price with Detected Jumps - JPM January 2022')
plt.xlabel('Tick Index')
plt.ylabel('Mid Price (USD)')
plt.legend()
plt.grid()
plt.tight_layout()
# plt.show()

survival_labels = pd.DataFrame({'z': z_values, 'delta': delta_values})
print(survival_labels.head(20))
print(survival_labels.describe())
print(f"Censored observations: {(survival_labels['delta'] == 0).sum()}")
print(f"Observed observations: {(survival_labels['delta'] == 1).sum()}")

survival_labels.to_csv('jpm_lm_survival_labels_january_2025_h1.csv', index=False)
print("Saved LM survival labels.")
