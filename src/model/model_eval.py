import numpy as np
import pandas as pd

import pickle
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import os
import logging

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# Check if logger is initialized successfully
try:
    logger.debug("Logger initialized successfully.")
except Exception as e:
    print("Logger initialization failed:", e)

def load_model(model_path):
    """Load the trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise

def load_test_data(test_data_path):
    """Load the test data from a CSV file."""
    try:
        test_data = pd.read_csv(test_data_path)
        X_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values
        logger.info(f"Test data loaded successfully from {test_data_path}")
        return X_test, y_test
    except Exception as e:
        logger.error(f"Failed to load test data from {test_data_path}: {e}")
        raise

def evaluate_model(clf, X_test, y_test):
    """Evaluate the model and calculate evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_probab = clf.predict_proba(X_test)[:, 1]
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_probab)
        }
        logger.info("Model evaluation completed successfully.")
        logger.debug(f"Evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        raise

def save_metrics(metrics, output_path):
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(output_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info(f"Metrics saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics to {output_path}: {e}")
        raise

def main():
    """Main function to execute the model evaluation pipeline."""
    model_path = '/Users/beni31/Documents/CampusX DSMP/2.0/cookie-cutter-pipeline/models/model.pkl'
    test_data_path = './data/features/test_bow.csv'
    metrics_output_path = 'metrics.json'
    try:
        clf = load_model(model_path)
        X_test, y_test = load_test_data(test_data_path)
        metrics = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics, metrics_output_path)
        logger.info("Model evaluation pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

# Execute the main function
if __name__ == "__main__":
    main()