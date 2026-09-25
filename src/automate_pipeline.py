from data_loading.pipeline import DataPipeline
from engine_models import model_operations

STOCKS_LIST = ["AAPL", "AMZN", "TSLA"]

def automate_pipeline():
    DEFAULT_PERIOD = "5y"
    print("AUTOMATIC PIPELINE STARTED")
    
    for ticker in STOCKS_LIST:
        try:
            pipeline = DataPipeline(ticker=ticker, period=DEFAULT_PERIOD,sequence_length=60)
            pipeline.data_pipeline()

        except Exception as e:
            print(f"ML pipeline failed for {ticker} with error: {str(e)}")

# def automate_model_training():
#     for ticker in STOCKS_LIST:
#         try:
#             model_operations.model_ops(ticker)
#         except Exception as e:
#             print(f"ML Model Training Operations failed for {ticker} with error: {str(e)}")
        