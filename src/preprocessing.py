"""
This script handles fetching, preprocessing, feature engineering, and annotation of stock market data
for SVM-based analysis.
"""

import os
import yfinance as yf
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_data(ticker, start_date, end_date):
    """
    Fetch historical stock market data for a given ticker symbol.
    """
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    save_path = os.path.join(DATA_DIR, f"{ticker}_raw.csv")
    df.to_csv(save_path)  # Save raw data to data folder
    print(f"Raw data saved to: {save_path}")
    return df

def generate_features(df):
    """
    Generate technical indicators as features for the SVM model.
    """
    print("Generating features...")

    # Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()

    # Bollinger Bands
    df['BB_upper'] = df['Close'].rolling(window=20).mean() + (df['Close'].rolling(window=20).std() * 2)
    df['BB_lower'] = df['Close'].rolling(window=20).mean() - (df['Close'].rolling(window=20).std() * 2)

    # Average True Range (ATR)
    df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()

    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Volatility (standard deviation of closing price)
    df['Volatility'] = df['Close'].rolling(window=10).std()

    # MACD (Moving Average Convergence Divergence)
    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema

    # Drop rows with NaN values from rolling calculations
    df.dropna(inplace=True)
    print("Feature generation complete.")
    return df




def handle_missing_values(df):
    """
    Handle missing values in the DataFrame.
    """
    print("Handling missing values...")
    df.ffill(inplace=True)  # Forward fill
    df.bfill(inplace=True)  # Backward fill
    return df

def normalize_data(df, features):
    """
    Normalize selected features using MinMaxScaler.
    """
    print("Normalizing data...")
    scaler = MinMaxScaler()
    for feature in tqdm(features, desc="Normalizing features"):
        df[feature] = scaler.fit_transform(df[[feature]])
    print("Normalization complete.")
    return df

def annotate_labels(df):
    """
    Annotate labels for classification based on next day's price movement.
    """
    print("Annotating labels...")
    
    # Flatten MultiIndex column names, if any
    if isinstance(df.columns, pd.MultiIndex):
        print("Flattening MultiIndex column names...")
        df.columns = ['_'.join(filter(None, col)) for col in df.columns]
        print(f"Flattened columns: {df.columns.tolist()}")

    # Dynamically detect the 'Close' column after flattening
    close_col = [col for col in df.columns if 'Close' in col]
    if not close_col:
        raise KeyError("No 'Close' column found in the DataFrame after flattening column names.")
    close_col = close_col[0]  # Use the first matching column
    print(f"Using '{close_col}' as the Close column.")

    # Create the Next_Close column by shifting the detected Close column
    df['Next_Close'] = df[close_col].shift(-1)
    print("Next_Close column created.")

    # Check for NaN values in 'Next_Close'
    nan_count = df['Next_Close'].isna().sum()
    print(f"Found {nan_count} rows with NaN in 'Next_Close'. Dropping them...")

    # Drop rows with NaN in 'Next_Close'
    df = df.dropna(subset=['Next_Close']).reset_index(drop=True)
    print(f"Rows after dropping NaN in 'Next_Close': {len(df)}")

    # Add the Label column
    df['Label'] = (df['Next_Close'] > df[close_col]).astype(int)  # 1 for favorable, 0 for unfavorable
    print("Label annotation complete.")
    
    # Debugging: Check the distribution of the labels
    print(f"Label distribution:\n{df['Label'].value_counts()}")
    return df

if __name__ == "__main__":
    # Configuration
    TICKER = "AAPL"  # Replace with your desired ticker
    START_DATE = "2020-01-01"
    END_DATE = "2023-01-01"

    # Fetch data
    raw_data = fetch_data(TICKER, START_DATE, END_DATE)

    # Handle missing values in the raw data
    raw_data = handle_missing_values(raw_data)

    # Generate features
    raw_data = generate_features(raw_data)

    # Normalize data for selected features
    features_to_normalize = ['SMA_10', 'SMA_30', 'RSI', 'Volatility', 'MACD']
    processed_data = normalize_data(raw_data, features_to_normalize)

    # Annotate labels
    labeled_data = annotate_labels(processed_data)

    # Save labeled data
    save_path = os.path.join(DATA_DIR, f"{TICKER}_labeled.csv")
    labeled_data.to_csv(save_path, index=False)
    print(f"Labeled data saved to: {save_path}")