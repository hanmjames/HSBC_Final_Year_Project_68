import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from lob_preprocessing import LOBPreprocessor, LOBDataset

preprocessor = LOBPreprocessor(stock="JPM", folder_name="jpm_data_2022_january")

msg_file = preprocessor.message_files[0]
ord_file = preprocessor.order_files[0]

message_data, orderbook_data = preprocessor.load_day(msg_file, ord_file)
print(f"Message data shape: {message_data.shape}")
print(f"Orderbook data shape: {orderbook_data.shape}")

features = preprocessor.compute_features(orderbook_data, message_data)
print(f"Features shape: {features.shape}")
print(f"Features columns: {features.columns.tolist()}")
print(f"Features head:\n{features.head()}")
print(f"Features describe:\n{features.describe()}")

normalized_features = preprocessor.run()
print(f"Normalized features shape: {normalized_features.shape}")
print(f"Number of possible windows: {len(normalized_features) - 100}")

dataset = LOBDataset(normalized_features, window_size=100)
print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Single window shape: {sample.shape}")