import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)

folder_name = "jpm_data_2022_january"

message_files = []
order_files = []

for file in sorted(os.listdir(folder_name)):
    if file.endswith("_message_10.csv") and file.startswith("JPM_2022-01"):
        message_files.append(file)
    elif file.endswith("_orderbook_10.csv") and file.startswith("JPM_2022-01"):
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

lstm_lookback_window = 50
theta = 3

def create_lstm_sequences(data, lookback_window):
    sequences = []
    for i in range(len(data) - lookback_window):
        sequences.append(data[i:i + lookback_window])
    return np.array(sequences)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

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
    all_mid_prices_nonzero.append(jpm_mid_price.iloc[jpm_log_returns.index])

    lstm_training_data = jpm_log_returns[:int(len(jpm_log_returns) * 0.8)]
    lstm_testing_data = jpm_log_returns[int(len(jpm_log_returns) * 0.8):]

    model = LSTMModel(input_size=1, hidden_size=20, num_layers=1, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    lstm_sequences = create_lstm_sequences(lstm_training_data.values, lstm_lookback_window)
    lstm_targets = lstm_training_data.values[lstm_lookback_window:]
    X_train = torch.FloatTensor(lstm_sequences).unsqueeze(-1)
    y_train = torch.FloatTensor(lstm_targets).unsqueeze(-1)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch == num_epochs - 1:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")

    model.eval()
    lstm_testing_sequences = create_lstm_sequences(lstm_testing_data.values, lstm_lookback_window)
    X_test = torch.FloatTensor(lstm_testing_sequences).unsqueeze(-1)
    y_test = torch.FloatTensor(lstm_testing_data.values[lstm_lookback_window:]).unsqueeze(-1)
    lstm_preds = model(X_test)
    lstm_residuals = y_test - lstm_preds.detach()
    lstm_residuals_standardized = (lstm_residuals - lstm_residuals.mean()) / lstm_residuals.std()

    jpm_jump_flag = (lstm_residuals_standardized.abs().squeeze() > theta).int()
    jpm_jump_indices = torch.where(jpm_jump_flag == 1)[0].numpy()
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

    all_jump_flags.append(pd.Series(jpm_jump_flag.numpy()))
    all_mid_prices.append(jpm_mid_price)
    all_log_returns.append(jpm_log_returns)
    all_timestamps.append(jpm_message_file[0][jpm_log_returns.index])

all_jump_flags = pd.concat(all_jump_flags, ignore_index=True)
all_mid_prices = pd.concat(all_mid_prices, ignore_index=True)
all_log_returns = pd.concat(all_log_returns, ignore_index=True)
all_timestamps = pd.concat(all_timestamps, ignore_index=True)
all_mid_prices_nonzero = pd.concat(all_mid_prices_nonzero, ignore_index=True)

# plt.figure(figsize=(15, 5))
# plt.plot(all_mid_prices_nonzero.values, linewidth=0.5, color='blue', label='Mid Price')
# jump_locations = all_jump_flags[all_jump_flags == 1].index
# plt.scatter(jump_locations, all_mid_prices_nonzero.iloc[jump_locations].values, color='red', s=30, zorder=5, label='Jumps', marker='x')
# plt.title('Mid Price with Detected Jumps (LSTM Residuals) - JPM January 2022')
# plt.xlabel('Tick Index')
# plt.ylabel('Mid Price (USD)')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

survival_labels = pd.DataFrame({'z': z_values, 'delta': delta_values})
print(f"Total ticks: {len(all_jump_flags)}")
print(f"Total jumps detected: {all_jump_flags.sum()}")
print(f"Jump rate: {all_jump_flags.mean():.4%}")
print(f"\nSurvival labels:")
print(survival_labels.head(20))
print(survival_labels.describe())
print(f"Censored observations: {(survival_labels['delta'] == 0).sum()}")
print(f"Observed observations: {(survival_labels['delta'] == 1).sum()}")
survival_labels.to_csv('jpm_lstm_residual_labels_january_2022.csv', index=False)
print("Saved LSTM residual survival labels.")