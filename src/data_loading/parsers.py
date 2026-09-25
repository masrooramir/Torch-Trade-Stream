import numpy as np
from sklearn.preprocessing import MinMaxScaler

class MarketDataProcessor:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range = (0,1)) 

    def scale_and_slice(self,raw_prices: np.ndarray):
        if len(raw_prices) < self.sequence_length+1:
            raise ValueError(f"Insufficient data. Got  {len(raw_prices)} points, need at least {self.sequence_length+1}")
        scaled_prices = self.scaler.fit_transform(raw_prices.reshape(-1,1))

        X,y = [], []
        for i in range(self.sequence_length, len(scaled_prices)):
            X.append(scaled_prices[i-self.sequence_length:i,0])
            y.append(scaled_prices[i,0])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


    def get_scaler_bounds(self):
        return np.array([self.scaler.min_, self.scaler.scale_], dtype=np.float32)