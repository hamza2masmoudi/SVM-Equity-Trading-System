import matplotlib.pyplot as plt
import pandas as pd

def plot_hyperparameter_results(cv_results):
    """
    Plots the results of hyperparameter tuning.
    """
    results = pd.DataFrame(cv_results)
    plt.figure(figsize=(10, 6))
    plt.plot(results['mean_test_score'], label='Mean Test Score', marker='o')
    plt.fill_between(
        range(len(results)),
        results['mean_test_score'] - results['std_test_score'],
        results['mean_test_score'] + results['std_test_score'],
        alpha=0.2
    )
    plt.xticks(range(len(results)), results.index, rotation=45)
    plt.xlabel('Parameter Combination')
    plt.ylabel('Accuracy')
    plt.title('Hyperparameter Tuning Results')
    plt.legend()
    plt.tight_layout()
    plt.show()