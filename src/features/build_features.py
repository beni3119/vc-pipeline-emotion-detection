# Ham data/processed se data leke aaenge or uspe BoW lagaenge
# Fir ham data ke andar ek naya folder bana denge data/features

import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import logging

logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)

try:
    max_features = yaml.safe_load(open('params.yaml', 'r'))['feature_engineering']['max_features']
    logger.info(f"Loaded max_features: {max_features}")
except Exception as e:
    logger.error(f"Error loading max_features from params.yaml: {e}")
    raise

def fetch_data():
    try:
        logger.info("Fetching processed train and test data...")
        train_data = pd.read_csv('./data/processed/train_processed.csv')
        test_data = pd.read_csv('./data/processed/test_processed.csv')
        logger.info("Data fetched successfully.")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise

try:
    train_data, test_data = fetch_data()
except Exception as e:
    logger.error("Failed to fetch data.")
    raise

def handle_missing_values(train_data, test_data):
    try:
        logger.debug("Handling missing values in train and test data...")
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        logger.info("Missing values handled.")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error handling missing values: {e}")
        raise

try:
    train_data, test_data = handle_missing_values(train_data, test_data)
except Exception as e:
    logger.error("Failed to handle missing values.")
    raise

def prepare_data(train_data, test_data):
    try:
        logger.debug("Preparing data for BoW transformation...")
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        logger.info("Data prepared for BoW.")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise

try:
    X_train, y_train, X_test, y_test = prepare_data(train_data, test_data)
except Exception as e:
    logger.error("Failed to prepare data.")
    raise

def apply_bow(X_train, X_test, max_features):
    try:
        logger.info("Applying Bag of Words (CountVectorizer)...")
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logger.info("BoW transformation complete.")
        return X_train_bow, X_test_bow
    except Exception as e:
        logger.error(f"Error applying BoW: {e}")
        raise

try:
    X_train_bow, X_test_bow = apply_bow(X_train, X_test, max_features)
except Exception as e:
    logger.error("Failed to apply BoW.")
    raise

def create_bow_dataframes(X_train_bow, y_train, X_test_bow, y_test):
    try:
        logger.debug("Creating DataFrames from BoW features...")
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.info("DataFrames created.")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error creating DataFrames: {e}")
        raise

try:
    train_df, test_df = create_bow_dataframes(X_train_bow, y_train, X_test_bow, y_test)
except Exception as e:
    logger.error("Failed to create DataFrames.")
    raise

def store_bow_data(train_df, test_df, data_path="data/features"):
    try:
        logger.info(f"Storing BoW data to {data_path}...")
        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)
        logger.info("BoW data stored successfully.")
    except Exception as e:
        logger.error(f"Error storing BoW data: {e}")
        raise

try:
    store_bow_data(train_df, test_df)
except Exception as e:
    logger.error("Failed to store BoW data.")
    raise