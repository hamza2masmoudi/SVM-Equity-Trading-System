import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/AAPL_labeled.csv")
TRAIN_PATH = os.path.join(BASE_DIR, "../data/train.csv")
TEST_PATH = os.path.join(BASE_DIR, "../data/test.csv")

# Load the labeled dataset
print("Loading labeled data...")
data = pd.read_csv(DATA_PATH)

# Separate features and labels
X = data.drop(columns=['Next_Close', 'Label'])
y = data['Label']

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine features and labels for saving
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save the training and testing sets
train_data.to_csv(TRAIN_PATH, index=False)
print(f"Training data saved to: {TRAIN_PATH}")

test_data.to_csv(TEST_PATH, index=False)
print(f"Testing data saved to: {TEST_PATH}")