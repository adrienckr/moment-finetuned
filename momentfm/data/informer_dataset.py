
# from typing import Optional
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# class InformerDataset:
#     def __init__(
#         self,
#         forecast_horizon: Optional[int] = 192,
#         data_split: str = "train",
#         data_stride_len: int = 1,
#         task_name: str = "forecasting",
#         random_seed: int = 42
#     ):
#         self.seq_len = 512
#         self.forecast_horizon = forecast_horizon
#         self.full_file_path_and_name = "train.csv"
#         self.data_split = data_split
#         self.data_stride_len = data_stride_len
#         self.task_name = task_name
#         self.random_seed = random_seed
#         # Read data
#         self._read_data()

#     def _get_borders(self):
#         n_train = 12 * 30 * 24
#         n_val = 4 * 30 * 24
#         n_test = 4 * 30 * 24
#         train_end = n_train
#         val_end = n_train + n_val
#         test_start = val_end - self.seq_len
#         test_end = test_start + n_test + self.seq_len
#         train = slice(0, train_end)
#         test = slice(test_start, test_end)
#         print(f"Train Slice: {train}, Test Slice: {test}")  # Debugging statement
#         return train, test

#     def _read_data(self):
#         self.scaler = StandardScaler()
#         df = pd.read_csv(self.full_file_path_and_name)
        
#         # Debug and ensure data shape is as expected
#         print(f"Original Data Shape: {df.shape}")
#         print(df.head())  # Check first rows
        
#         self.length_timeseries_original = df.shape[0]
        
#         # Assuming 'sales' is the target variable we are forecasting
#         # Removing non-numeric columns or encoding them if needed
#         df.drop(columns=["id", "date", "store_nbr", "family"], inplace=True)
        
#         # Handling missing values and scaling the remaining data
#         df = df.infer_objects(copy=False).interpolate(method="cubic")
#         data_splits = self._get_borders()
        
#         # Splitting data
#         train_data = df.iloc[data_splits[0]]
#         print(f"Train Data Shape: {train_data.shape}")  # Debugging statement
#         self.scaler.fit(train_data.values)
#         df = self.scaler.transform(df.values)
        
#         if self.data_split == "train":
#             self.data = df[data_splits[0], :]
#         elif self.data_split == "test":
#             self.data = df[data_splits[1], :]
        
#         self.length_timeseries = self.data.shape[0]
#         print(f"Processed Data Shape: {self.data.shape}")  # Debugging statement

#     def __getitem__(self, index):
#         seq_start = self.data_stride_len * index
#         seq_end = seq_start + self.seq_len

#         # Handle if seq_start is negative after adjustment
#         seq_start = max(0, seq_start)
#         seq_end = min(self.length_timeseries, seq_end)

#         print(f"Seq Start: {seq_start}, Seq End: {seq_end}")  # Debugging statement
        
#         input_mask = np.ones(self.seq_len)
        
#         if self.task_name == "forecasting":
#             pred_end = seq_end + self.forecast_horizon

#             seq_end = seq_end - self.forecast_horizon
#             seq_start = max(0, seq_end - self.seq_len)

#             pred_end = min(self.length_timeseries, pred_end)
            
#             print(f"Adjusted Seq Start: {seq_start}, Seq End: {seq_end}, Pred End: {pred_end}")  # Debugging statement
#             timeseries = self.data[seq_start:seq_end, :].T
#             forecast = self.data[seq_end:pred_end, :].T

#             print(f"Timeseries Shape: {timeseries.shape}, Forecast Shape: {forecast.shape}")  # Debugging statement
#             return timeseries, forecast, input_mask
        
#         elif self.task_name == "imputation":
#             if seq_end > self.length_timeseries:
#                 seq_end = self.length_timeseries
#             seq_end = seq_end - self.seq_len
#             timeseries = self.data[seq_start:seq_end, :].T
#             print(f"Imputation Timeseries Shape: {timeseries.shape}")  # Debugging statement
#             return timeseries, input_mask

#     def __len__(self):
#         if self.task_name == "imputation":
#             return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
#         elif self.task_name == "forecasting":
#             return (
#                 self.length_timeseries - self.seq_len - self.forecast_horizon
#             ) // self.data_stride_len + 1


# dataset = InformerDataset()


# from typing import Optional
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# class InformerDataset:
#     def __init__(
#         self,
#         forecast_horizon: Optional[int] = 192,
#         data_split: str = "train",
#         data_stride_len: int = 512,
#         task_name: str = "forecasting",
#         random_seed: int = 42,
#     ):
#         self.seq_len = 512
#         self.forecast_horizon = forecast_horizon
#         self.full_file_path_and_name = "train.csv"  # Update with your path
#         self.data_split = data_split
#         self.data_stride_len = data_stride_len
#         self.task_name = task_name
#         self.random_seed = random_seed

#         self._read_data()

#     # def _get_borders(self):
#     #     total_length = self.length_timeseries_original
#     #     # Use 80% of data for training, 20% for testing
#     #     n_train = int(0.9 * total_length)
#     #     test_start = n_train - self.seq_len  # Ensure enough history for test sequences
#     #     test_end = total_length

#     #     train_slice = slice(0, n_train)
#     #     test_slice = slice(test_start, test_end)
#     #     return train_slice, test_slice
#     def _get_borders(self):
#         # Use percentage-based splits instead of fixed calendar periods
#         total = self.length_timeseries_original
#         n_train = int(0.8 * total)  # 80% for training
#         test_start = n_train - self.seq_len  # Ensure test sequences have history
#         test_end = total  # Use remaining 20% for testing
        
#         return slice(0, n_train), slice(test_start, test_end)

#     # def _read_data(self):
#     #     self.scaler = StandardScaler()
#     #     df = pd.read_csv(self.full_file_path_and_name)
        
#     #     # Process data: aggregate sales by date
#     #     df['date'] = pd.to_datetime(df['date'])
#     #     df = df.groupby('date')['sales'].sum().reset_index()
        
#     #     # Handle missing dates and interpolate
#     #     dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
#     #     df = df.set_index('date').reindex(dates).rename(columns={'sales': 'sales'}).reset_index()
#     #     df['sales'] = df['sales'].interpolate(method='linear')
        
#     #     self.length_timeseries_original = len(df)
#     #     self.n_channels = 1  # Univariate (sales only)

#     #     # Prepare data (exclude date column)
#     #     data = df[['sales']].values
        
#     #     # Split data
#     #     train_slice, test_slice = self._get_borders()
        
#     #     # Scale data using training split
#     #     self.scaler.fit(data[train_slice])
#     #     data = self.scaler.transform(data)
        
#     #     if self.data_split == "train":
#     #         self.data = data[train_slice]
#     #     elif self.data_split == "test":
#     #         self.data = data[test_slice]
        
#     #     self.length_timeseries = len(self.data)

#     def _read_data(self):
#         self.scaler = StandardScaler()
#         df = pd.read_csv(self.full_file_path_and_name)
        
#         # Convert to datetime and remove time component
#         df['date'] = pd.to_datetime(df['date']).dt.normalize()  # <- Key fix
        
#         # Aggregate sales by date (now truly daily)
#         df = df.groupby('date')['sales'].sum().reset_index()
        
#         # Handle missing dates
#         dates = pd.date_range(
#             start=df['date'].min(), 
#             end=df['date'].max(), 
#             freq='D'
#         )
        
#         # Create clean daily index
#         df = (
#             df.set_index('date')
#             .reindex(dates)
#             .rename(columns={'sales': 'sales'})
#             .reset_index()
#         )
        
#         # Interpolate missing values
#         df['sales'] = df['sales'].interpolate(method='linear')
        
#         # Rest of your code remains the same
#         self.length_timeseries_original = len(df)
#         self.n_channels = 1
    
#         train_slice, test_slice = self._get_borders()
        
#         self.scaler.fit(df.loc[train_slice, ['sales']].values)
#         scaled_data = self.scaler.transform(df[['sales']].values)
        
#         if self.data_split == "train":
#             self.data = scaled_data[train_slice]
#         elif self.data_split == "test":
#             self.data = scaled_data[test_slice]
        
#         self.length_timeseries = len(self.data)

    
#     def __getitem__(self, index):
#         seq_start = self.data_stride_len * index
#         seq_end = seq_start + self.seq_len
#         input_mask = np.ones(self.seq_len)

#         if self.task_name == "forecasting":
#             pred_end = seq_end + self.forecast_horizon
#             # Handle cases where sequence exceeds data bounds
#             if pred_end > self.length_timeseries:
#                 pred_end = self.length_timeseries
#                 seq_end = pred_end - self.forecast_horizon
#                 seq_start = seq_end - self.seq_len

#             # Extract sequences (shape: [n_channels, seq_len])
#             timeseries = self.data[seq_start:seq_end].T  # (1, seq_len)
#             forecast = self.data[seq_end:pred_end].T     # (1, forecast_horizon)

#             return timeseries.astype(np.float32), forecast.astype(np.float32), input_mask

#         elif self.task_name == "imputation":
#             if seq_end > self.length_timeseries:
#                 seq_end = self.length_timeseries
#                 seq_start = seq_end - self.seq_len

#             timeseries = self.data[seq_start:seq_end].T
#             return timeseries.astype(np.float32), input_mask

#     def __len__(self):
#         if self.task_name == "imputation":
#             return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
#         elif self.task_name == "forecasting":
#             return (
#                 self.length_timeseries - self.seq_len - self.forecast_horizon
#             ) // self.data_stride_len + 1




from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
class InformerDataset:
    def __init__(
        self,
        forecast_horizon: Optional[int] = 192,
        data_split: str = "train",
        data_stride_len: int = 1,
        task_name: str = "forecasting",
        random_seed: int = 42,
    ):
        self.seq_len = 512
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = "train.csv"
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed

        self._read_data()

    def _get_borders(self):
        total = self.length_timeseries_original
        n_train = int(0.8 * total)
        test_start = n_train - self.seq_len
        test_end = total
        return slice(0, n_train), slice(test_start, test_end)

    def _read_data(self):
        self.scaler = StandardScaler()
        
        # Load and preprocess data
        df = pd.read_csv(self.full_file_path_and_name, parse_dates=['date'])
        
        # Aggregate sales by date and handle missing dates
        df = (
            df.groupby('date')['sales'].sum()  # Aggregate sales
            .reindex(pd.date_range(df['date'].min(), df['date'].max(), freq='D'))
            .rename('sales').reset_index()
            .interpolate(method='linear')
        )
        
        self.length_timeseries_original = len(df)
        self.n_channels = 1  # Single channel for sales

        # Split and scale
        train_slice, test_slice = self._get_borders()
        train_data = df.iloc[train_slice]['sales'].values.reshape(-1, 1)
        
        self.scaler.fit(train_data)
        scaled_data = self.scaler.transform(df['sales'].values.reshape(-1, 1))
        
        if self.data_split == "train":
            self.data = scaled_data[train_slice]
        else:
            self.data = scaled_data[test_slice]

        self.length_timeseries = len(self.data)

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        
        if self.task_name == "forecasting":
            pred_end = seq_end + self.forecast_horizon
            
            # Handle edge cases
            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = pred_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            # Return shapes: (1, seq_len), (1, horizon)
            return (
                torch.FloatTensor(self.data[seq_start:seq_end].T),  # (1, 512)
                torch.FloatTensor(self.data[seq_end:pred_end].T),   # (1, 192)
                torch.ones(self.seq_len)  # Input mask
            )

    def __len__(self):
        return (
            self.length_timeseries - self.seq_len - self.forecast_horizon
        ) // self.data_stride_len + 1