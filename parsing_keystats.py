import pandas as pd
import os
import time
import re
from datetime import datetime
from utils import data_string_to_float
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# The directory where individual html files are stored
statspath = os.getenv("STATSPATH", "intraQuarter/_KeyStats/")

# The list of features to parse from the html files
features = [
    "Market Cap", "Enterprise Value", "Trailing P/E", "Forward P/E", "PEG Ratio",
    "Price/Sales", "Price/Book", "Enterprise Value/Revenue", "Enterprise Value/EBITDA",
    "Profit Margin", "Operating Margin", "Return on Assets", "Return on Equity",
    "Revenue", "Revenue Per Share", "Qtrly Revenue Growth", "Gross Profit",
    "EBITDA", "Net Income Avl to Common", "Diluted EPS", "Qtrly Earnings Growth",
    "Total Cash", "Total Cash Per Share", "Total Debt", "Total Debt/Equity",
    "Current Ratio", "Book Value Per Share", "Operating Cash Flow", "Levered Free Cash Flow",
    "Beta", "50-Day Moving Average", "200-Day Moving Average", "Avg Vol (3 month)",
    "Shares Outstanding", "Float", "% Held by Insiders", "% Held by Institutions",
    "Shares Short (as of", "Short Ratio", "Short % of Float", "Shares Short (prior month"
]

def load_data(file_path):
    """
    Load data from a CSV file and drop any rows with missing values
    :param file_path: Path to the CSV file
    :return: DataFrame with data
    """
    try:
        data_df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        data_df.dropna(axis=0, how="any", inplace=True)
        return data_df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        return pd.DataFrame()

def reindex_and_fill(df, start_date, end_date):
    """
    Reindex the dataframe to include all days in the date range and fill missing values
    :param df: DataFrame to be reindexed
    :param start_date: Start date for the reindexing
    :param end_date: End date for the reindexing
    :return: Reindexed DataFrame with filled missing values
    """
    idx = pd.date_range(start_date, end_date)
    df = df.reindex(idx)
    df.ffill(inplace=True)
    return df

def preprocess_price_data():
    """
    Preprocess SP500 and stock price datasets to fill missing rows
    :return: SP500 and stock dataframes with no missing rows
    """
    sp500_df = load_data("sp500_index.csv")
    stock_df = load_data("stock_prices.csv")

    if sp500_df.empty or stock_df.empty:
        logging.error("Error loading SP500 or stock data. Exiting preprocessing.")
        return pd.DataFrame(), pd.DataFrame()

    start_date = str(stock_df.index[0])
    end_date = str(stock_df.index[-1])

    sp500_df = reindex_and_fill(sp500_df, start_date, end_date)
    stock_df = reindex_and_fill(stock_df, start_date, end_date)

    return sp500_df, stock_df

def parse_html_file(file_path, features):
    """
    Parse an HTML file to extract feature values
    :param file_path: Path to the HTML file
    :param features: List of features to extract
    :return: List of extracted feature values
    """
    value_list = []
    try:
        with open(file_path, "r") as file:
            source = file.read().replace(",", "")
            for variable in features:
                try:
                    regex = (
                        r">" + re.escape(variable) + r".*?(\-?\d+\.*\d*K?M?B?|N/A[\\n|\s]*|>0|NaN)%?"
                        r"(</td>|</span>)"
                    )
                    value = re.search(regex, source, flags=re.DOTALL).group(1)
                    value_list.append(data_string_to_float(value))
                except AttributeError:
                    if variable == "Avg Vol (3 month)":
                        try:
                            new_variable = ">Average Volume (3 month)"
                            regex = (
                                re.escape(new_variable) + r".*?(\-?\d+\.*\d*K?M?B?|N/A[\\n|\s]*|>0)%?"
                                r"(</td>|</span>)"
                            )
                            value = re.search(regex, source, flags=re.DOTALL).group(1)
                            value_list.append(data_string_to_float(value))
                        except AttributeError:
                            value_list.append("N/A")
                    else:
                        value_list.append("N/A")
    except Exception as e:
        logging.error(f"Error parsing HTML file {file_path}: {str(e)}")
        value_list = ["N/A"] * len(features)
    return value_list

def parse_keystats(sp500_df, stock_df):
    """
    Parse key statistics from HTML files and create a dataset
    :param sp500_df: DataFrame containing SP500 prices
    :param stock_df: DataFrame containing stock prices
    :return: DataFrame of parsed key statistics and stock performance data
    """
    stock_list = [x[0] for x in os.walk(statspath)][1:]

    df_columns = [
        "Date", "Unix", "Ticker", "Price", "stock_p_change", "SP500", "SP500_p_change"
    ] + features
    df = pd.DataFrame(columns=df_columns)

    for stock_directory in tqdm(stock_list, desc="Parsing progress:", unit="tickers"):
        keystats_html_files = os.listdir(stock_directory)
        if ".DS_Store" in keystats_html_files:
            keystats_html_files.remove(".DS_Store")

        ticker = stock_directory.split(statspath)[1]

        for file in keystats_html_files:
            date_stamp = datetime.strptime(file, "%Y%m%d%H%M%S.html")
            unix_time = time.mktime(date_stamp.timetuple())
            full_file_path = os.path.join(stock_directory, file)

            value_list = parse_html_file(full_file_path, features)

            current_date = datetime.fromtimestamp(unix_time).strftime("%Y-%m-%d")
            one_year_later = datetime.fromtimestamp(unix_time + 31536000).strftime("%Y-%m-%d")

            try:
                sp500_price = float(sp500_df.loc[current_date, "Adj Close"])
                sp500_1y_price = float(sp500_df.loc[one_year_later, "Adj Close"])
                sp500_p_change = round(((sp500_1y_price - sp500_price) / sp500_price * 100), 2)
            except KeyError:
                logging.warning(f"SP500 data missing for {current_date} or {one_year_later}")
                continue

            try:
                stock_price = float(stock_df.loc[current_date, ticker.upper()])
                stock_1y_price = float(stock_df.loc[one_year_later, ticker.upper()])
                stock_p_change = round(((stock_1y_price - stock_price) / stock_price * 100), 2)
            except KeyError:
                logging.warning(f"Stock data missing for {ticker}