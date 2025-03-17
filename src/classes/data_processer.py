import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, minmax_scale
import torch
from torch.utils.data import DataLoader, TensorDataset


class DataProcesser:
    def __init__(self):
        pass

    @staticmethod
    def prepare_tensor_for_lstm(data_series, sequence_length, device="cpu"):
        """
        Converts a Pandas Series into a PyTorch tensor formatted for LSTM input.

        Args:
            data_series (pd.Series): Time series data to be formatted.

        Returns:
            torch.Tensor: A tensor ready for LSTM input with shape (batch_size, seq_length, input_size).
        """
        if not isinstance(data_series, pd.Series):
            raise ValueError("Input must be a Pandas Series.")

        # Convert to NumPy array
        data_np = data_series.to_numpy()

        # Create overlapping sequences using a sliding window approach
        num_samples = len(data_np) - sequence_length + 1
        if num_samples <= 0:
            raise ValueError(f"Sequence length {sequence_length} is too large for the data.")

        tensor_data = np.array([data_np[i:i + sequence_length] for i in range(num_samples)])

        # Convert to PyTorch tensor and reshape for LSTM (batch_size, seq_length, input_size)
        tensor_data = torch.tensor(tensor_data, dtype=torch.float32).unsqueeze(-1)  # Add input_size dimension

        # Move tensor to the correct device
        tensor_data = tensor_data.to(device)

        return tensor_data

    def normalize_series(self, series, distribution = 'lognorm'):

        if distribution == 'lognorm':
            return series.apply(np.log)
        else:
            raise NotImplementedError('Method not yet implemented!')

    def prepare_autoregressive_data(self, series, window_size=10, stride=1, train_ratio=0.7, val_ratio=0.15, batch_size=32, test=False, scaler_method = 'standard'):
        """
        Converts a time series into supervised learning format (X, y) and splits it into train, validation, and test sets.
        
        Args:
            series (pd.Series): Time series data.
            window_size (int): Number of past time steps to use as input.
            stride (int): Step size for windowing.
            train_ratio (float): Proportion of data used for training.
            val_ratio (float): Proportion of data used for validation.
            batch_size (int): Batch size for DataLoaders.
            test (bool): If True, return only test_loader with the entire series and the scaler.

        Returns:
            If test=True:
                DataLoader: Test DataLoader.
                StandardScaler: The fitted scaler.
            Otherwise:
                DataLoader: Train DataLoader.
                DataLoader: Validation DataLoader.
                DataLoader: Test DataLoader.
                StandardScaler: The fitted scaler.
        """
        if scaler_method == 'standard':
            # Normalize series using StandardScaler
            scaler = StandardScaler()
            series_scaled = scaler.fit_transform(series.values.reshape(-1, 1))
        elif scaler_method == 'minmax':
            # Scale series using minmax_scale
            scaler = None
            series_scaled = minmax_scale(series.values.reshape(-1, 1))
        else:
            raise NotImplemented(f"{scaler} method is not yet implemented!")
        series_tensor = torch.tensor(series_scaled, dtype=torch.float32)

        # Create input-target pairs using sliding window
        X, y = [], []
        for i in range(0, len(series_tensor) - window_size, stride):
            X.append(series_tensor[i:i + window_size])
            y.append(series_tensor[i + window_size])  # Next time step as target

        X_tensor = torch.stack(X)
        y_tensor = torch.stack(y)

        # If test mode, return the entire dataset as test_loader
        if test:
            test_dataset = TensorDataset(X_tensor, y_tensor)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            return None, None, test_loader, scaler

        # Determine split indices
        total_samples = X_tensor.shape[0]
        train_end = int(train_ratio * total_samples)
        val_end = train_end + int(val_ratio * total_samples)

        # Split data into train, validation, and test sets
        X_train, y_train = X_tensor[:train_end], y_tensor[:train_end]
        X_val, y_val = X_tensor[train_end:val_end], y_tensor[train_end:val_end]
        X_test, y_test = X_tensor[val_end:], y_tensor[val_end:]

        # Create PyTorch datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, scaler
    
    def process_ICE_data(self, df: pd.DataFrame, series_name: list[str]):
        
        for series in series_name:
            df[series] = self.normalize_series(df[series], 'lognorm')
        train_loader, val_loader, test_loader, scaler = self.prepare_autoregressive_data(df[series_name])

        return df, train_loader, val_loader, test_loader, scaler