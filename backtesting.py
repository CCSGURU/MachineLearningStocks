import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from utils import status_calc
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def split_data(data_df, test_size=0.2):
    """
    Split the dataset into train and test sets
    :param data_df: DataFrame with data
    :param test_size: Proportion of the dataset to include in the test split
    :return: Split datasets (X_train, X_test, y_train, y_test, z_train, z_test)
    """
    features = data_df.columns[6:]
    X = data_df[features].values
    y = list(status_calc(data_df["stock_p_change"], data_df["SP500_p_change"], outperformance=10))
    z = np.array(data_df[["stock_p_change", "SP500_p_change"]])
    return train_test_split(X, y, z, test_size=test_size, random_state=0)

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model
    :param X_train: Training data features
    :param y_train: Training data labels
    :return: Trained model
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    """
    Evaluate the model's performance
    :param clf: Trained model
    :param X_test: Test data features
    :param y_test: Test data labels
    :return: Predicted labels and evaluation metrics
    """
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    return y_pred, accuracy, precision

def calculate_returns(y_pred, z_test):
    """
    Calculate the returns for the predicted stocks
    :param y_pred: Predicted labels
    :param z_test: Test data returns
    :return: Stock and market returns, and performance metrics
    """
    num_positive_predictions = sum(y_pred)
    if num_positive_predictions <= 0:
        logging.warning("No stocks predicted!")
        return 0, 0, 0

    stock_returns = 1 + z_test[y_pred, 0] / 100
    market_returns = 1 + z_test[y_pred, 1] / 100
    avg_predicted_stock_growth = sum(stock_returns) / num_positive_predictions
    index_growth = sum(market_returns) / num_positive_predictions
    percentage_stock_returns = 100 * (avg_predicted_stock_growth - 1)
    percentage_market_returns = 100 * (index_growth - 1)
    total_outperformance = percentage_stock_returns - percentage_market_returns

    return num_positive_predictions, percentage_stock_returns, percentage_market_returns, total_outperformance

def backtest():
    """
    Perform a simple backtest on the dataset
    :return: None
    """
    logging.info("Starting backtest...")
    data_df = load_data("keystats.csv")
    if data_df.empty:
        logging.error("Data loading failed. Exiting backtest.")
        return

    X_train, X_test, y_train, y_test, z_train, z_test = split_data(data_df)
    clf = train_model(X_train, y_train)
    y_pred, accuracy, precision = evaluate_model(clf, X_test, y_test)

    logging.info("Classifier performance\n%s", "=" * 20)
    logging.info(f"Accuracy score: {accuracy:.2f}")
    logging.info(f"Precision score: {precision:.2f}")

    num_positive_predictions, percentage_stock_returns, percentage_market_returns, total_outperformance = calculate_returns(y_pred, z_test)

    logging.info("\nStock prediction performance report\n%s", "=" * 40)
    logging.info(f"Total Trades: {num_positive_predictions}")
    logging.info(f"Average return for stock predictions: {percentage_stock_returns:.1f}%")
    logging.info(f"Average market return in the same period: {percentage_market_returns:.1f}%")
    logging.info(f"Compared to the index, our strategy earns {total_outperformance:.1f} percentage points more")

if __name__ == "__main__":
    backtest()