import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import logging
import time
from typing import List
from functools import wraps
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

stocks = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'JPM', 'JNJ', 'V', 'PG',
    'XOM', 'KO', 'INTC', 'PFE', 'CSCO', 'T', 'WMT', 'CVX', 'CRM', 'NFLX',
    'NVDA', 'MA', 'DIS', 'BAC', 'ADBE', 'NKE', 'MCD', 'LLY', 'TXN', 'BMY',
    'UNH', 'QCOM', 'MDT', 'IBM', 'AMGN', 'LOW', 'CAT', 'BA', 'HON', 'SBUX',
    'GILD', 'BLK', 'BKNG', 'SPGI', 'SYK', 'VRTX', 'SPY'  # Added for benchmarking
]
bonds = ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'BND']
commodities = ['GLD', 'SLV', 'USO', 'DBC']
sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU', 'XLB', 'XLRE']
currencies = ['UUP', 'FXE', 'FXY', 'FXB']
cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD']
all_tickers = stocks + bonds + commodities + sectors + currencies + cryptos

start_date = '2015-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry a function on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempts} failed: {str(e)}. Retrying...")
                    time.sleep(delay * (2 ** attempts))
        return wrapper
    return decorator

@retry(max_attempts=3, delay=1.0)
def download_batch(batch: List[str], start: str, end: str) -> pd.DataFrame:
    """Download data for a batch of tickers."""
    df = yf.download(batch, start=start, end=end, auto_adjust=True, threads=True)['Close']
    if df.empty:
        logger.warning(f"No data downloaded for batch: {batch}")
    return df.astype(np.float32)

def download_and_save(tickers: List[str], start: str, end: str, filename: str, batch_size: int = 10, sleep_sec: float = 1.0) -> None:
    """Download and save asset data in batches."""
    if os.path.exists(filename):
        logger.info(f"File '{filename}' already exists. Skipping download.")
        return
    
    all_data = pd.DataFrame()
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        logger.info(f"Downloading batch: {batch}...")
        try:
            df = download_batch(batch, start, end)
            if not df.empty:
                all_data = pd.concat([all_data, df], axis=1)
            else:
                logger.warning(f"No data for batch: {batch}")
        except Exception as e:
            logger.warning(f"Skipping batch {batch} due to error: {str(e)}")
        time.sleep(sleep_sec)
    
    if not all_data.empty:
        all_data = all_data.loc[:, ~all_data.columns.duplicated()]
        all_data.to_csv(filename)
        logger.info(f"Data saved to {filename}")
    else:
        logger.error("No data downloaded. Check ticker list or network connection.")

if __name__ == "__main__":
    os.makedirs("historical_data", exist_ok=True)
    filename = "historical_data/close_assets.csv"
    download_and_save(all_tickers, start_date, end_date, filename)