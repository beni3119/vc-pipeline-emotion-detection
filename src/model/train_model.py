import numpy as np
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import GradientBoostingClassifier
import logging

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# Check if logger is initialized
try:
    logger.debug("Logger initialized successfully.")
except Exception as e:
    print("Logger initialization failed:", e)

try:
    params = yaml.safe_load(open('params.yaml','r'))['model_building']
    logger.info("Parameters loaded from params.yaml successfully.")
except Exception as e:
    logger.error(f"Failed to load parameters from params.yaml: {e}")
    raise

def load_data(file_path):
    try:
        train_data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}.")
        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values
        return X_train, y_train
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def train_model(X_train, y_train, params):
    try:
        clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
        clf.fit(X_train, y_train)
        logger.info("Model trained successfully.")
        return clf
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def save_model(model, file_path):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully to {file_path}.")
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}")
        raise

# Main execution
try:
    X_train, y_train = load_data('./data/features/train_bow.csv')
    clf = train_model(X_train, y_train, params)
    save_model(clf, 'models/model.pkl')
except Exception as e:
    logger.error(f"Pipeline execution failed: {e}")
