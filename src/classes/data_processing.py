import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, minmax_scale
import torch
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.seasonal import STL
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer


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
    
    def prepare_autoencoder_data(self, series, window_size=10, stride=1, train_ratio=0.7, val_ratio=0.15, batch_size=32, test=False, scaler_method='standard'):
        """
        Prepares time series data for LSTM Autoencoder training, where X == Y (reconstruction task).

        Args:
            series (pd.Series): Time series data.
            window_size (int): Length of each input sequence.
            stride (int): Step size for sliding window.
            train_ratio (float): Fraction of data used for training.
            val_ratio (float): Fraction of data used for validation.
            batch_size (int): Size of batches in DataLoader.
            test (bool): If True, returns only the test set.
            scaler_method (str): 'standard' or 'minmax' scaling.

        Returns:
            If test=True:
                test_loader, scaler
            Else:
                train_loader, val_loader, test_loader, scaler
        """
        from sklearn.preprocessing import StandardScaler, minmax_scale
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        # Scale the series
        if scaler_method == 'standard':
            scaler = StandardScaler()
            if len(series.shape) == 2:
                print(series.shape)
                series_scaled = scaler.fit_transform(series.values)
            else:
                series_scaled = scaler.fit_transform(series.values.reshape(-1, 1))
        elif scaler_method == 'minmax':
            scaler = None
            if len(series.shape) == 2:
                series_scaled = minmax_scale(series.values)
            else:
                series_scaled = minmax_scale(series.values.reshape(-1, 1))
        else:
            raise NotImplementedError(f"{scaler_method} method is not yet implemented!")

        series_tensor = torch.tensor(series_scaled, dtype=torch.float32)

        # Extract sequences
        X = []
        for i in range(0, len(series_tensor) - window_size + 1, stride):
            window = series_tensor[i:i + window_size]
            X.append(window)

        X_tensor = torch.stack(X)  # shape: (n_samples, window_size, 1)

        # Autoencoder target is same as input
        Y_tensor = X_tensor.clone()

        if test:
            test_dataset = TensorDataset(X_tensor, Y_tensor)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            return None, None, test_loader, scaler

        # Split indices
        total_samples = X_tensor.shape[0]
        train_end = int(train_ratio * total_samples)
        val_end = train_end + int(val_ratio * total_samples)

        # Split sets
        X_train, Y_train = X_tensor[:train_end], Y_tensor[:train_end]
        X_val, Y_val = X_tensor[train_end:val_end], Y_tensor[train_end:val_end]
        X_test, Y_test = X_tensor[val_end:], Y_tensor[val_end:]

        # Build datasets
        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)
        test_dataset = TensorDataset(X_test, Y_test)

        # Build loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, scaler
    
    def process_ICE_data(self, df: pd.DataFrame, series_name: list[str]):
        
        for series in series_name:
            df[series] = self.normalize_series(df[series], 'lognorm')
        train_loader, val_loader, test_loader, scaler = self.prepare_autoregressive_data(df[series_name])

        return df, train_loader, val_loader, test_loader, scaler
    
class SignalProcesser:

    def __init__(self):
        pass

    def detrend(self, signal, period, method = 'stl'):

        if method.lower() == 'stl':
            stl = STL(signal, period = period)
            res = stl.fit()
            detrended_signal = signal - res.trend
        elif method.lower() == 'differencing':
            detrended_signal = np.diff(signal)
        else:
            raise NotImplementedError(f"Method: {method} is not yet implemented!")

        return detrended_signal
    
    def log_transform(self, signal):

        signal_log = np.log(signal[signal > 0]) # avoid log(0)

        return signal_log
    
    def check_normality_plot(self, signal):
        
        stats.probplot(signal, dist = "norm", plot = plt)

    def quantile_transformation(self, signal):

        qt = QuantileTransformer(output_distribution='normal')
        data_normalized = qt.fit_transform(signal.reshape(-1, 1)).flatten()

        return data_normalized