import numpy as np
import pandas as pd
import math
import os
import torch

class LOBDataset(torch.utils.data.Dataset):
    def __init__(self, features, window_size = 100):
        self.features = features
        self.window_size = window_size

    #returns number of possible windows
    def __len__(self):
        return len(self.features) - self.window_size
    
    #returns one particular window when called
    def __getitem__(self, index):
        return self.features[index:index + self.window_size]

class LOBPreprocessor:
    def __init__(self, stock, folder_name, window_size = 100):
        self.stock = stock
        self.folder_name = folder_name
        self.window_size = window_size
        self.message_files = []
        self.order_files = []
        self.load_files()

    def load_files(self):
        for file in sorted(os.listdir(self.folder_name)):
            if file.endswith("_message_10.csv") and file.startswith(self.stock + "_2022-01"):
                self.message_files.append(file)
            elif file.endswith("_orderbook_10.csv") and file.startswith(self.stock + "_2022-01"):
                self.order_files.append(file)

    def load_day(self, msg_file, ord_file):
        message_data = pd.read_csv(os.path.join(self.folder_name, msg_file), header=None, low_memory=False)
        orderbook_data = pd.read_csv(os.path.join(self.folder_name, ord_file), header=None, low_memory=False)
        
        orderbook_data.replace([9999999999, -9999999999, 0], float('nan'), inplace=True)
        
        mask = (message_data[0] >= 34800) & (message_data[0] <= 57000)
        message_data = message_data[mask].reset_index(drop=True)
        orderbook_data = orderbook_data[mask].reset_index(drop=True)
        
        return message_data, orderbook_data

    def compute_features(self, orderbook_data, message_data):
        ask_price = orderbook_data[[0, 4, 8, 12, 16, 20, 24, 28, 32, 36]] / 10000
        ask_size = orderbook_data[[1, 5, 9, 13, 17, 21, 25, 29, 33, 37]]
        bid_price = orderbook_data[[2, 6, 10, 14, 18, 22, 26, 30, 34, 38]] / 10000
        bid_size = orderbook_data[[3, 7, 11, 15, 19, 23, 27, 31, 35, 39]]
        timestamp = message_data[0]

        bid_ask_spreads = pd.DataFrame()
        mid_prices = pd.DataFrame()

        for i in range(10):
            bid_ask_spreads[f"spread_{i+1}"] = ask_price.iloc[:, i] - bid_price.iloc[:, i]
            mid_prices[f"mid_price_{i+1}"] = (ask_price.iloc[:, i] + bid_price.iloc[:, i]) / 2

        ask_price_difference = ask_price.diff(axis = 1).fillna(0)
        bid_price_difference = bid_price.diff(axis = 1).fillna(0)

        mean_ask_price = ask_price.mean(axis = 1)
        mean_bid_price = bid_price.mean(axis = 1)
        mean_ask_size = ask_size.mean(axis = 1)
        mean_bid_size = bid_size.mean(axis = 1)
        mean_bid_ask_spread = bid_ask_spreads.mean(axis = 1)
        mean_vol_imbalance = (ask_size - bid_size).mean(axis = 1)

        ask_price_derivatives = ask_price.iloc[:, 0].diff().fillna(0) / timestamp.diff()
        ask_size_derivatives = ask_size.iloc[:, 0].diff().fillna(0) / timestamp.diff()
        bid_price_derivatives = bid_price.iloc[:, 0].diff().fillna(0) / timestamp.diff()
        bid_size_derivatives = bid_size.iloc[:, 0].diff().fillna(0) / timestamp.diff()

        delta_t = 10 #hyperparameter for arrival intensity calculation

        message_data["is_limit_ask"] = ((message_data[1] == 1) & (message_data[5] == -1)).astype(int)
        message_data["is_limit_bid"] = ((message_data[1] == 1) & (message_data[5] == 1)).astype(int)
        message_data["is_market_ask"] = ((message_data[1].isin([4, 5])) & (message_data[5] == -1)).astype(int)
        message_data["is_market_bid"] = ((message_data[1].isin([4, 5])) & (message_data[5] == 1)).astype(int)
        message_data["is_cancel_ask"] = ((message_data[1].isin([2,3])) & (message_data[5] == -1)).astype(int)
        message_data["is_cancel_bid"] = ((message_data[1].isin([2,3])) & (message_data[5] == 1)).astype(int)

        message_data = message_data.set_index(pd.to_datetime(message_data[0], unit='s'))

        limit_ask_arrival_intensity = message_data["is_limit_ask"].rolling(f'{delta_t}s').sum() / delta_t
        limit_bid_arrival_intensity = message_data["is_limit_bid"].rolling(f'{delta_t}s').sum() / delta_t
        market_ask_arrival_intensity = message_data["is_market_ask"].rolling(f'{delta_t}s').sum() / delta_t
        market_bid_arrival_intensity = message_data["is_market_bid"].rolling(f'{delta_t}s').sum() / delta_t
        cancel_ask_arrival_intensity = message_data["is_cancel_ask"].rolling(f'{delta_t}s').sum() / delta_t
        cancel_bid_arrival_intensity = message_data["is_cancel_bid"].rolling(f'{delta_t}s').sum() / delta_t

        delta_T = 60 #longer historical window for v8

        limit_ask_arrival_avg = limit_ask_arrival_intensity.rolling(f'{delta_T}s').mean()
        limit_bid_arrival_avg = limit_bid_arrival_intensity.rolling(f'{delta_T}s').mean()
        market_ask_arrival_avg = market_ask_arrival_intensity.rolling(f'{delta_T}s').mean()
        market_bid_arrival_avg = market_bid_arrival_intensity.rolling(f'{delta_T}s').mean()
        cancel_ask_arrival_avg = cancel_ask_arrival_intensity.rolling(f'{delta_T}s').mean()
        cancel_bid_arrival_avg = cancel_bid_arrival_intensity.rolling(f'{delta_T}s').mean()

        limit_ask_relative_intensity = (limit_ask_arrival_intensity > limit_ask_arrival_avg).astype(int)
        limit_bid_relative_intensity = (limit_bid_arrival_intensity > limit_bid_arrival_avg).astype(int)
        market_ask_relative_intensity = (market_ask_arrival_intensity > market_ask_arrival_avg).astype(int)
        market_bid_relative_intensity = (market_bid_arrival_intensity > market_bid_arrival_avg).astype(int)
        cancel_ask_relative_intensity = (cancel_ask_arrival_intensity > cancel_ask_arrival_avg).astype(int)
        cancel_bid_relative_intensity = (cancel_bid_arrival_intensity > cancel_bid_arrival_avg).astype(int)

        time_diff = pd.Series(limit_ask_arrival_intensity.index).diff().dt.total_seconds().fillna(1).values
        time_diff = np.clip(time_diff, 1e-6, None)

        limit_ask_acceleration = np.clip(limit_ask_arrival_intensity.diff().fillna(0).values / time_diff, -1e6, 1e6)
        limit_bid_acceleration = np.clip(limit_bid_arrival_intensity.diff().fillna(0).values / time_diff, -1e6, 1e6)
        market_ask_acceleration = np.clip(market_ask_arrival_intensity.diff().fillna(0).values / time_diff, -1e6, 1e6)
        market_bid_acceleration = np.clip(market_bid_arrival_intensity.diff().fillna(0).values / time_diff, -1e6, 1e6)
        cancel_ask_acceleration = np.clip(cancel_ask_arrival_intensity.diff().fillna(0).values / time_diff, -1e6, 1e6)
        cancel_bid_acceleration = np.clip(cancel_bid_arrival_intensity.diff().fillna(0).values / time_diff, -1e6, 1e6)

        clock_time = np.floor(timestamp / 60).astype(int)

        ask_price_reset = ask_price.reset_index(drop=True)
        ask_size_reset = ask_size.reset_index(drop=True)
        bid_price_reset = bid_price.reset_index(drop=True)
        bid_size_reset = bid_size.reset_index(drop=True)
        spreads_reset = bid_ask_spreads.reset_index(drop=True)
        mid_prices_reset = mid_prices.reset_index(drop=True)
        ask_price_reset.columns = [f'ask_price_{i+1}' for i in range(10)]
        ask_size_reset.columns = [f'ask_size_{i+1}' for i in range(10)]
        bid_price_reset.columns = [f'bid_price_{i+1}' for i in range(10)]
        bid_size_reset.columns = [f'bid_size_{i+1}' for i in range(10)]

        ask_price_difference_reset = ask_price_difference.reset_index(drop=True)
        bid_price_difference_reset = bid_price_difference.reset_index(drop=True)
        ask_price_difference_reset.columns = [f'ask_price_diff_{i+1}' for i in range(10)]
        bid_price_difference_reset.columns = [f'bid_price_diff_{i+1}' for i in range(10)]

        features = pd.concat([
            ask_price_reset,
            ask_size_reset,
            bid_price_reset,
            bid_size_reset,
            spreads_reset,
            mid_prices_reset,
            ask_price_difference_reset,
            bid_price_difference_reset,
            pd.DataFrame({
                "mean_ask_price": mean_ask_price.reset_index(drop=True),
                "mean_bid_price": mean_bid_price.reset_index(drop=True),
                "mean_ask_size": mean_ask_size.reset_index(drop=True),
                "mean_bid_size": mean_bid_size.reset_index(drop=True),
                "mean_bid_ask_spread": mean_bid_ask_spread.reset_index(drop=True),
                "mean_vol_imbalance": mean_vol_imbalance.reset_index(drop=True),
                "ask_price_derivative": pd.Series(ask_price_derivatives).reset_index(drop=True),
                "bid_price_derivative": pd.Series(bid_price_derivatives).reset_index(drop=True),
                "ask_size_derivative": pd.Series(ask_size_derivatives).reset_index(drop=True),
                "bid_size_derivative": pd.Series(bid_size_derivatives).reset_index(drop=True),
                "limit_ask_arrival_intensity": limit_ask_arrival_intensity.reset_index(drop=True),
                "limit_bid_arrival_intensity": limit_bid_arrival_intensity.reset_index(drop=True),
                "market_ask_arrival_intensity": market_ask_arrival_intensity.reset_index(drop=True),
                "market_bid_arrival_intensity": market_bid_arrival_intensity.reset_index(drop=True),
                "cancel_ask_arrival_intensity": cancel_ask_arrival_intensity.reset_index(drop=True),
                "cancel_bid_arrival_intensity": cancel_bid_arrival_intensity.reset_index(drop=True),
                "limit_ask_relative_intensity": limit_ask_relative_intensity.reset_index(drop=True),
                "limit_bid_relative_intensity": limit_bid_relative_intensity.reset_index(drop=True),
                "market_ask_relative_intensity": market_ask_relative_intensity.reset_index(drop=True),
                "market_bid_relative_intensity": market_bid_relative_intensity.reset_index(drop=True),
                "cancel_ask_relative_intensity": cancel_ask_relative_intensity.reset_index(drop=True),
                "cancel_bid_relative_intensity": cancel_bid_relative_intensity.reset_index(drop=True),
                "limit_ask_acceleration": pd.Series(limit_ask_acceleration).reset_index(drop=True),
                "limit_bid_acceleration": pd.Series(limit_bid_acceleration).reset_index(drop=True),
                "market_ask_acceleration": pd.Series(market_ask_acceleration).reset_index(drop=True),
                "market_bid_acceleration": pd.Series(market_bid_acceleration).reset_index(drop=True),
                "cancel_ask_acceleration": pd.Series(cancel_ask_acceleration).reset_index(drop=True),
                "cancel_bid_acceleration": pd.Series(cancel_bid_acceleration).reset_index(drop=True),
                "clock_time": pd.Series(clock_time).reset_index(drop=True),
            })
        ], axis=1)

        return features
    
    def normalize(self, curr_features, prev_features): #prev_features is a list of feature dataframes of prev 5 days' data
        combined_features = pd.concat(prev_features)
        combined_mean = combined_features.mean()
        combined_std = combined_features.std()

        normalized_features = (curr_features - combined_mean) / combined_std

        return normalized_features 
    
    def run(self):
        all_features = []
        all_normalized_features = []
        
        for i, (msg_file, ord_file) in enumerate(zip(self.message_files, self.order_files)):
            print(f"Processing day {i+1}/{len(self.message_files)}: {msg_file}")
            message_data, orderbook_data = self.load_day(msg_file, ord_file)
            features = self.compute_features(orderbook_data, message_data)
            all_features.append(features)
            
            if i < 5:
                continue
            
            prev_features = all_features[i-5:i]
            normalized_features = self.normalize(features, prev_features) 
            all_normalized_features.append(normalized_features)

        all_normalized_features_combined = pd.concat(all_normalized_features, ignore_index=True)
        all_normalized_features_combined.to_parquet(f'{self.stock}_normalized_features.parquet')     
        return all_normalized_features_combined
