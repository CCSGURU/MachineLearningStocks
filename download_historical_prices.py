import os
import logging
from pandas_datareader import data as pdr
import pandas as pd
import fix_yahoo_finance as yf

# Override Yahoo Finance API
yf.pdr_override()

# Constants for start and end dates
START_DATE = os.getenv("START_DATE", "2003-08-01")
END_DATE = os.getenv("END_DATE", "2015-01-01")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_ticker_list(statspath):
    """
    Retrieves the list of tickers from the specified directory.
    :param statspath: Path to the directory containing ticker files
    :return: List of tickers
    """
    ticker_list = os.listdir(statspath)
    if ".DS_Store" in ticker_list:
        os.remove(os.path.join(statspath, ".DS_Store"))
        ticker_list.remove(".DS_Store")
    return ticker_list

def download_data(ticker_list, start, end):
    """
    Downloads stock data for the given tickers between the specified dates.
    :param ticker_list: List of tickers
    :param start: Start date
    :param end: End date
    :return: DataFrame of stock data
    """
    try:
        all_data = pdr.get_data_yahoo(ticker_list, start, end)
        stock_data = all_data["Adj Close"]
        stock_data.dropna(how="all", axis=1, inplace=True)
        stock_data.ffill(inplace=True)
        return stock_data
    except Exception as e:
        logging.error("Error downloading data: %s", str(e))
        return pd.DataFrame()

def build_stock_dataset(start=START_DATE, end=END_DATE):
    """
    Creates the dataset containing all stock prices.
    :returns: None
    """
    logging.info("Building stock dataset...")
    statspath = "intraQuarter/_KeyStats/"
    ticker_list = get_ticker_list(statspath)
    stock_data = download_data(ticker_list, start, end)
    if not stock_data.empty:
        stock_data.to_csv("stock_prices.csv")
        missing_tickers = [ticker for ticker in ticker_list if ticker.upper() not in stock_data.columns]
        logging.info("%d tickers are missing: %s", len(missing_tickers), missing_tickers)
    else:
        logging.warning("No stock data available.")

def build_sp500_dataset(start=START_DATE, end=END_DATE):
    """
    Creates the dataset containing S&P500 prices.
    :returns: None
    """
    logging.info("Building S&P500 dataset...")
    try:
        index_data = pdr.get_data_yahoo("SPY", start, end)
        index_data.to_csv("sp500_index.csv")
    except Exception as e:
        logging.error("Error downloading S&P500 data: %s", str(e))

def build_dataset_iteratively(idx_start, idx_end, date_start=START_DATE, date_end=END_DATE):
    """
    Alternative iterative solution to building the stock dataset.
    :param idx_start: Starting index of the ticker list
    :param idx_end: Ending index of the ticker list
    :param date_start: Start date for data download
    :param date_end: End date for data download
    :returns: None
    """
    logging.info("Building stock dataset iteratively...")
    statspath = "intraQuarter/_KeyStats/"
    ticker_list = get_ticker_list(statspath)[idx_start:idx_end]
    df = pd.DataFrame()
    for ticker in ticker_list:
        ticker = ticker.upper()
        try:
            stock_ohlc = pdr.get_data_yahoo(ticker, start=date_start, end=date_end)
            if stock_ohlc.empty:
                logging.warning("No data for %s", ticker)
                continue
            adj_close = stock_ohlc["Adj Close"].rename(ticker)
            df = pd.concat([df, adj_close], axis=1)
        except Exception as e:
            logging.error("Error downloading data for %s: %s", ticker, str(e))
    df.to_csv("stock_prices.csv")

if __name__ == "__main__":
    build_stock_dataset()
    build_sp500_dataset()