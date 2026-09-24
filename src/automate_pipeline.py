from scraper.pipeline import DataPipeline

def automate():
    WATCHLIST = ["AAPL", "AMZN", "TSLA"]
    DEFAULT_PERIOD = "5y"
    
    print("🤖 AUTOMATED CRON PIPELINE STARTED")
    
    for ticker in WATCHLIST:
        try:
            pipeline = DataPipeline(ticker=ticker, sequence_length=60)
            pipeline.run_historical_pipeline(period=DEFAULT_PERIOD)
        except Exception as e:
            print(f"❌ Failed to sync {ticker}: {str(e)}")
    print("🤖 AUTOMATED CRON PIPELINE FINISHED SUCCESSFULLY")