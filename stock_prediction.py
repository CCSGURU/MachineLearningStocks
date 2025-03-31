import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import data_string_to_float, status_calc
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# The percentage by which a stock has to beat the S&P500 to be considered a 'buy'
OUTPERFORMANCE = int(os.getenv("OUTPERFORMANCE", 10))

def load_data(file_path):
    """
    Load data from a CSV file and drop any rows with missing values
    :param file_path: Path to the CSV file
    :return: DataFrame with data
    """
    try:
        data_df = pd.read_csv(file_path, index_col="Date")
        data_df.dropna(axis=0, how="any", inplace=True)
        return data_df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        return pd.DataFrame()

def build_data_set():
    """
    Reads the keystats.csv file and prepares it for scikit-learn
    :return: X_train and y_train numpy arrays
    """
    training_data = load_data("keystats.csv")
    if training_data.empty:
        logging.error("Training data loading failed. Exiting build_data_set.")
        return None, None
    
    features = training_data.columns[6:]
    X_train = training_data[features].values
    y_train = list(
        status_calc(
            training_data["stock_p_change"],
            training_data["