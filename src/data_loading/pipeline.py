import os, sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import yfinance as yf
import pandas as pd
import numpy as np
from data_loading.parsers import MarketDataProcessor
from config.settings import PROCESSED_DIR , RAW_DIR,init_workspace

class DataPipeline:
    def __init__(self, ticker: str = "AAPL", period: str = "5y", sequence_length: int = 60):
        init_workspace()
        self.period = period
        self.ticker = ticker.upper()
        self.raw_path = os.path.join(RAW_DIR, f"{self.ticker}_raw.csv")
        self.processor = MarketDataProcessor(sequence_length=sequence_length)


    def data_pipeline(self):
        self._data_fetching_historical_pipeline()
        self._data_Processing_historical_pipeline()


    def _data_fetching_historical_pipeline(self):
        # Data Fetching
        print(f"\n \nDATA FETCHING PIPELINE FOR {self.ticker} HAS STARTED")
        ticker_obj = yf.Ticker(self.ticker)
        raw_df = ticker_obj.history(period=self.period, interval="1d")

        if raw_df.empty:
            raise ValueError(f"Extracted payload for {self.ticker} returned empty. Aborting.")
        raw_df.to_csv(self.raw_path)


    def _data_Processing_historical_pipeline(self):
        # Feature Engineering
        df = pd.read_csv(self.raw_path)
        close_prices = pd.to_numeric(df['Close'], errors='coerce').dropna().values
        X, y = self.processor.scale_and_slice(close_prices) 

        # Train-Test Split
        split_perc = int(len(X) * 0.8)
        X_train, X_test = X[:split_perc], X[split_perc:]
        y_train, y_test = y[:split_perc], y[split_perc:]

        # Saving Pre-processed Data
        np.save(os.path.join(PROCESSED_DIR, f"{self.ticker}_X_train.npy"), X_train)
        np.save(os.path.join(PROCESSED_DIR, f"{self.ticker}_X_test.npy"), X_test)
        np.save(os.path.join(PROCESSED_DIR, f"{self.ticker}_y_train.npy"), y_train)
        np.save(os.path.join(PROCESSED_DIR, f"{self.ticker}_y_test.npy"), y_test)
        
        np.save(os.path.join(PROCESSED_DIR, f"{self.ticker}_scaler_params.npy"), self.processor.get_scaler_bounds())
        
        print(f"DATA FETCHING PIPELINE SUCCESSFUL \nTrain, Test Data Shape: {X_train.shape} -- {X_test.shape}")