import numpy as np
import pandas as pd
import yaml
import os

from sklearn.model_selection import train_test_split

import logging

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
            logger.info(f"Loaded test_size={test_size} from {params_path}")
            return test_size
    except FileNotFoundError:
        logger.error(f"The file {params_path} does not exist.")
        raise
    except KeyError as e:
        logger.error(f"Missing key in the parameters file: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise ValueError(f"Error parsing YAML file: {e}")

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logger.debug(f"Data read successfully from {url}")
        return df
    except FileNotFoundError:
        logger.error(f"The file at {url} does not exist.")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"The file at {url} is empty.")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing the CSV file at {url}: {e}")
        raise ValueError(f"Error parsing the CSV file at {url}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading the file at {url}: {e}")
        raise RuntimeError(f"An unexpected error occurred while reading the file at {url}: {e}")

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if 'sentiment' not in df.columns:
            logger.error("The required column 'sentiment' is missing from the DataFrame.")
            raise KeyError("The required column 'sentiment' is missing from the DataFrame.")
        
        final_df = df[df['sentiment'].isin(['neutral', 'sadness'])]
        final_df['sentiment'].replace({'neutral': 1, 'sadness': 0}, inplace=True)
        logger.debug("Data processed successfully.")
        return final_df
    except KeyError as e:
        logger.error(f"Error processing data: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing the data: {e}")
        raise RuntimeError(f"An unexpected error occurred while processing the data: {e}")

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.debug(f"Train and test data saved to {data_path}")
    except PermissionError:
        logger.error(f"Permission denied: Unable to write to directory {data_path}.")
        raise
    except FileNotFoundError:
        logger.error(f"The specified path {data_path} does not exist.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving data: {e}")
        raise RuntimeError(f"An unexpected error occurred while saving data: {e}")

def main() -> None:
    test_size = load_params('params.yaml')
    df = read_data("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv")
    final_df = process_data(df)
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

    data_path = os.path.join("data", "raw")
    save_data(data_path, train_data, test_data)

if __name__ == "__main__":
    main()






