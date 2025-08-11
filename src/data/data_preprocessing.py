#Data/raw folder me ghuske train or test wali files ko uthaenge or unpe transformation apply karenge
# Fir ham data wale folder ke andar ek or folder banenge processed bolke jisme train.processed or test.processed files banenge

import numpy as np
import pandas as pd

import os
import logging
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
  logger.addHandler(console_handler)

def fetch_data():
  try:
    logger.info("Fetching train and test data from data/raw directory.")
    train_data = pd.read_csv('./data/raw/train.csv')  # Assuming the train data is in this path
    test_data = pd.read_csv('./data/raw/test.csv')    # Assuming the test data is in this path
    logger.debug("Train and test data loaded successfully.")
    return train_data, test_data
  except Exception as e:
    logger.error(f"Error fetching data: {e}")
    raise

try:
  train_data, test_data = fetch_data()
except Exception as e:
  logger.error("Failed to fetch data. Exiting script.")
  raise

def lemmatization(text):
  lemmatizer = WordNetLemmatizer()
  text = text.split()
  text = [lemmatizer.lemmatize(y) for y in text]
  return " ".join(text)

def remove_stop_words(text):
  stop_words = set(stopwords.words("english"))
  Text = [i for i in str(text).split() if i not in stop_words]
  return " ".join(Text)

def removing_numbers(text):
  text = ''.join([i for i in text if not i.isdigit()])
  return text

def lower_case(text):
  text = text.split()
  text = [y.lower() for y in text]
  return " ".join(text)

def removing_punctuations(text):
  # Removing punctuations
  text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]&_`{|}~"""), ' ', text)
  text = text.replace(':', "", )
  # removing extra white-spaces
  text = re.sub('\s+', ' ', text)
  text = " ".join(text.split())
  return text.strip()

def removing_urls(text):
  url_pattern = re.compile(r'https?://\S+|www\.\S+')
  return url_pattern.sub(r'', text)

def remove_small_sentences(df):
  for i in range(len(df)):
    if len(df.text.iloc[i].split()) < 3:
      df.text.iloc[i] = np.nan

def normalize_text(df):
  try:
    logger.info("Normalizing text data.")
    df.content = df.content.apply(lambda content: lower_case(content))
    df.content = df.content.apply(lambda content: remove_stop_words(content))
    df.content = df.content.apply(lambda content: removing_numbers(content))
    df.content = df.content.apply(lambda content: removing_punctuations(content))
    df.content = df.content.apply(lambda content: removing_urls(content))
    df.content = df.content.apply(lambda content: lemmatization(content))
    logger.debug("Text normalization completed.")
    return df
  except Exception as e:
    logger.error(f"Error during text normalization: {e}")
    raise

try:
  train_processed_data = normalize_text(train_data)
  test_processed_data = normalize_text(test_data)
except Exception as e:
  logger.error("Failed to normalize data. Exiting script.")
  raise

def save_processed_data(train_data, test_data, output_dir="data/processed"):
  try:
    logger.info(f"Saving processed data to {output_dir}")
    # Create the processed data directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Save the processed train and test data
    train_data.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)
    logger.debug("Processed data saved successfully.")
  except Exception as e:
    logger.error(f"Error saving processed data: {e}")
    raise

# Call the function to save the processed data
try:
  save_processed_data(train_processed_data, test_processed_data)
except Exception as e:
  logger.error("Failed to save processed data. Exiting script.")
  raise