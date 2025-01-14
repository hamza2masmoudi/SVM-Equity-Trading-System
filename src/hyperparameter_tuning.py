import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "../data/train.csv")

# Load training data
print("Loading training data...")
train_data = pd.read_csv(TRAIN_PATH)

# Separate features and labels
X_train = train_data.drop(columns=['Label'])
y_train = train_data['Label']

# Apply SMOTE for balancing the dataset
print("Balancing dataset with SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['linear', 'rbf']
}

# Perform GridSearchCV
print("Starting hyperparameter tuning with GridSearchCV...")
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train_balanced, y_train_balanced)

# Print best parameters
print("Best Parameters:", grid_search.best_params_)