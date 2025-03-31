import pandas as pd
import os
import re
import time
import requests
import numpy as np
from tqdm import tqdm
from utils import data_string_to_float
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# The path to your fundamental data
statspath = os.getenv("STATSPATH", "intraQuarter/_KeyStats/")
forwardpath = os.getenv("FORWARDPATH", "forward/")

# These are the features that will be parsed
features = [
    "Market Cap", "Enterprise Value", "Trailing P/E", "Forward P/E", "PEG Ratio",
    "Price/Sales", "Price/Book", "Enterprise Value/Revenue", "Enterprise Value/EBITDA",
    "Profit Margin", "Operating Margin", "Return on Assets", "Return on Equity",
    "Revenue", "Revenue Per Share", "Quarterly Revenue Growth", "Gross Profit",
    "EBITDA", "Net Income Avi to Common", "Diluted EPS", "Quarterly Earnings Growth",
    "Total Cash", "Total Cash Per Share", "Total Debt", "Total Debt/Equity",
    "Current Ratio", "Book Value Per Share", "Operating Cash Flow", "Levered Free Cash Flow",
    "Beta", "50-Day Moving Average", "200-Day Moving Average", "Avg Vol (3 month)",
    "Shares Outstanding", "Float", "% Held by Insiders", "% Held by Institutions",
    "Shares Short", "Short Ratio", "Short % of Float", "Shares Short (prior month)"
]

def get_ticker_list(statspath):
    """
    Retrieves the list of tickers from the specified directory.
    :param statspath: Path to the directory containing ticker files
    :return: List of tickers
    """
    ticker_list = os.listdir(statspath)
    if ".DS_Store" in ticker_list:
        ticker_list.remove(".DS_Store")
    return ticker_list

def download_html(ticker):
    """
    Downloads the HTML file for a given ticker from Yahoo Finance.
    :param ticker: Stock ticker symbol
    """
    try:
        link = f"http://finance.yahoo.com/quote/{ticker.upper()}/key-statistics"
        resp = requests.get(link)
        save_path = os.path.join(forwardpath, f"{ticker}.html")
        with open(save_path, "w") as file:
            file.write(resp.text)
    except Exception as e:
        logging.error(f"Error downloading data for {ticker}: {str(e)}")
        time.sleep(2)

def check_yahoo():
    """
    Retrieves the stock ticker from the _KeyStats directory, then downloads the HTML file from Yahoo Finance.
    :return: None
    """
    if not os.path.exists(forwardpath):
        os.makedirs(forwardpath)

    ticker_list = get_ticker_list(statspath)

    for ticker in tqdm(ticker_list, desc="Download progress:", unit="tickers"):
        download_html(ticker)

def parse_html(tickerfile):
    """
    Parses the HTML file to extract feature values.
    :param tickerfile: HTML file for a ticker
    :return: List of feature values
    """
    try:
        with open(os.path.join(forwardpath, tickerfile)) as file:
            source = file.read().replace(",", "")
            
        value_list = []
        for variable in features:
            regex = (
                r">" + re.escape(variable) + r".*?(\-?\d+\.*\d*K?M?B?|N/A[\\n|\s]*|>0|NaN)%?"
                r"(</td>|</span>)"
            )
            match = re.search(regex, source, flags=re.DOTALL)
            value = match.group(1) if match else "N/A"
            value_list.append(data_string_to_float(value))
        return value_list
    except Exception as e:
        logging.error(f"Error parsing file {tickerfile}: {str(e)}")
        return ["N/A"] * len(features)

def forward():
    """
    Creates the forward sample by parsing the current data HTML files that we downloaded in check_yahoo().
    :return: pandas DataFrame containing all of the current data for each ticker.
    """
    df_columns = [
        "Date", "Unix", "Ticker", "Price", "stock_p_change", "SP500", "SP500_p_change"
    ] + features

    df = pd.DataFrame(columns=df_columns)
    tickerfile_list = os.listdir(forwardpath)

    if ".DS_Store" in tickerfile_list:
        tickerfile_list.remove(".DS_Store")

    for tickerfile in tqdm(tickerfile_list, desc="Parsing progress:", unit="tickers"):
        ticker = tickerfile.split(".html")[0].upper()
        value_list = parse_html(tickerfile)
        new_df_row = [0, 0, ticker, 0, 0, 0, 0] + value_list
        df = df.append(dict(zip(df_columns, new_df_row)), ignore_index=True)

    return df.replace("N/A", np.nan)

if __name__ == "__main__":
    check_yahoo()
    current_df = forward()
    current_df.to_csv("forward_sample.csv", index=False)