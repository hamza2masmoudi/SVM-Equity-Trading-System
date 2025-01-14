import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from scipy.stats import uniform

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

# Define the parameter distribution
param_distributions = {
    'C': uniform(0.1, 10),  # Sample C from a uniform distribution
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['linear', 'rbf']
}

# Perform RandomizedSearchCV
print("Starting hyperparameter tuning with RandomizedSearchCV...")
random_search = RandomizedSearchCV(
    estimator=SVC(random_state=42),
    param_distributions=param_distributions,
    n_iter=10,  # Number of parameter combinations to try
    cv=5,       # 5-fold cross-validation
    scoring='accuracy',
    verbose=2,  # Print progress for each combination
    n_jobs=-1,  # Use all available cores
    random_state=42
)
random_search.fit(X_train_balanced, y_train_balanced)

# Print best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best Accuracy Score:", random_search.best_score_)

# Evaluate on the training data
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_train_balanced)
final_accuracy = accuracy_score(y_train_balanced, y_pred)
print("Final Accuracy on Balanced Training Data:", final_accuracy)