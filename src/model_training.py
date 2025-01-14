import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
import joblib


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "../data/train.csv")
TEST_PATH = os.path.join(BASE_DIR, "../data/test.csv")
SVM_MODEL_PATH = os.path.join(BASE_DIR, "../models/svm_model.pkl")
RF_MODEL_PATH = os.path.join(BASE_DIR, "../models/rf_model.pkl")
RESULTS_PATH = os.path.join(BASE_DIR, "../results/evaluation_report.txt")

# Load training and testing data
print("Loading training and testing data...")
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

# Separate features and labels
X_train = train_data.drop(columns=['Label'])
y_train = train_data['Label']
X_test = test_data.drop(columns=['Label'])
y_test = test_data['Label']

# Apply SMOTE for balancing the dataset
print("Balancing dataset with SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Initialize the SVM model
print("Initializing SVM model...")
svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42, probability=True)

# Train the SVM model with a progress bar
print("Training SVM model...")
for _ in tqdm(range(1), desc="Training Progress"):
    svm_model.fit(X_train_balanced, y_train_balanced)

# Evaluate the SVM model
print("Evaluating SVM model...")
svm_pred = svm_model.predict(X_test)
svm_report = classification_report(y_test, svm_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)

# Save SVM results
print(f"Saving SVM evaluation report to: {RESULTS_PATH}")
with open(RESULTS_PATH, "w") as file:
    file.write("SVM Classification Report:\n")
    file.write(svm_report)
    file.write(f"\nSVM Accuracy: {svm_accuracy:.2f}\n")

# Save the trained SVM model
print(f"Saving SVM model to: {SVM_MODEL_PATH}")
joblib.dump(svm_model, SVM_MODEL_PATH)

# Initialize the Random Forest model
print("Initializing Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model
print("Training Random Forest model...")
rf_model.fit(X_train_balanced, y_train_balanced)

# Evaluate the Random Forest model
print("Evaluating Random Forest model...")
rf_pred = rf_model.predict(X_test)
rf_report = classification_report(y_test, rf_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Save RF results
with open(RESULTS_PATH, "a") as file:
    file.write("\nRandom Forest Classification Report:\n")
    file.write(rf_report)
    file.write(f"\nRandom Forest Accuracy: {rf_accuracy:.2f}\n")

# Save the trained Random Forest model
print(f"Saving Random Forest model to: {RF_MODEL_PATH}")
joblib.dump(rf_model, RF_MODEL_PATH)

print("Model training and evaluation complete.")