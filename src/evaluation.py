import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(BASE_DIR, "../data/test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../models/svm_model.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "../results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# File paths for saving results
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
ROC_CURVE_PATH = os.path.join(RESULTS_DIR, "roc_curve.png")
EVALUATION_REPORT_PATH = os.path.join(RESULTS_DIR, "evaluation_report.txt")

# Load test data and model
print("Loading test data and model...")
test_data = pd.read_csv(TEST_PATH)
svm_model = joblib.load(MODEL_PATH)

# Separate features and labels
X_test = test_data.drop(columns=['Label'])
y_test = test_data['Label']

# Make predictions
print("Evaluating model...")
y_pred = svm_model.predict(X_test)
y_prob = svm_model.decision_function(X_test)  # Decision scores for ROC curve

# Classification report and accuracy
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save evaluation report
with open(EVALUATION_REPORT_PATH, "w") as file:
    file.write("Classification Report:\n")
    file.write(report)
    file.write(f"\nAccuracy: {accuracy:.2f}\n")
print(f"Evaluation report saved to: {EVALUATION_REPORT_PATH}")

# Confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(CONFUSION_MATRIX_PATH)
plt.close()
print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

# ROC curve
print("Generating ROC curve...")
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig(ROC_CURVE_PATH)
plt.close()
print(f"ROC curve saved to: {ROC_CURVE_PATH}")

print("Evaluation complete.")