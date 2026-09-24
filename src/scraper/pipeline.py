import os
import yfinance as yf
import pandas as pd
import numpy as np
from scraper.parsers import MarketDataProcessor

# DIR Locations
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Create DIR if missing
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


class DataPipeline:

    def __init__(self, ticker: str = "AAPL", sequence_length: int = 60):
        self.ticker = ticker.upper()
        self.raw_path = os.path.join(RAW_DATA_DIR, f"{self.ticker}_raw.csv")
        self.processor = MarketDataProcessor(sequence_length=sequence_length)

    def run_historical_pipeline(self, period: str = "2y"):        

        
        # Data Fetching
        print(f"INITIALIZING HISTORICAL PIPELINE FOR {self.ticker}")
        print(f"Raw Data Fetching for {period} of historical daily indices via Ticker Engine...")
        ticker_obj = yf.Ticker(self.ticker)
        raw_df = ticker_obj.history(period=period, interval="1d")
        if raw_df.empty:
            raise ValueError(f"Extracted payload for {self.ticker} returned empty. Aborting.")
        raw_df.to_csv(self.raw_path)
        print(f"✔ Saved clean raw backup data block to: {self.raw_path}")



        # PARSING & FEATURE ENGINEERING
        print("[2/4] Executing feature engineering and scaling calculations...")
        df = pd.read_csv(self.raw_path)
        
        # Pull values, explicitly force conversion to numeric, and drop NaNs
        close_prices = pd.to_numeric(df['Close'], errors='coerce').dropna().values
        
        X, y = self.processor.scale_and_slice(close_prices)

        # Train-Test Split (Training-80% , Testing-20%)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]


        # Saving Data in .npy binaries
        print(f"[3/4] Writing zero-latency binary files to {PROCESSED_DATA_DIR}...")
        np.save(os.path.join(PROCESSED_DATA_DIR, f"{self.ticker}_X_train.npy"), X_train)
        np.save(os.path.join(PROCESSED_DATA_DIR, f"{self.ticker}_X_test.npy"), X_test)
        np.save(os.path.join(PROCESSED_DATA_DIR, f"{self.ticker}_y_train.npy"), y_train)
        np.save(os.path.join(PROCESSED_DATA_DIR, f"{self.ticker}_y_test.npy"), y_test)
        
        np.save(os.path.join(PROCESSED_DATA_DIR, f"{self.ticker}_scaler_params.npy"), self.processor.get_scaler_bounds())
        
        print("\nDATA PIPELINE SUCCESSFUL")
        print(f"Training Shape: {X_train.shape}")
        print(f"Testing Shape: {X_test.shape}")

if __name__ == "__main__":
    pipeline = DataPipeline(ticker="AAPL", sequence_length=60)
    pipeline.run_historical_pipeline()