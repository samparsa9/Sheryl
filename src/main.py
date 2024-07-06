import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CSV_DIRECTORY, FRED_KEY, GCS_BUCKET_NAME, SENDER_EMAIL, EMAIL_PASSWORD
from src.data_collection.sp500_scraper import create_sp500_csv
from src.data_collection.crypto_scraper import create_crypto_csv
from src.data_collection.calculate_features import calculate_features
from src.data_processing.scaler import scale_data
from src.data_processing.k_means_cluster import apply_k_means
from src.utils.gcs_utils import upload_to_gcs
from src.utils.email_utils import send_email

def main():
    os.makedirs(CSV_DIRECTORY, exist_ok=True)
    crypto = True

    if crypto:
        create_crypto_csv(os.path.join(CSV_DIRECTORY, 'crypto_df.csv'))
        df = pd.read_csv(os.path.join(CSV_DIRECTORY, 'crypto_df.csv'), index_col='Symbol')
    else:
        create_sp500_csv(os.path.join(CSV_DIRECTORY, 'sp500_df.csv'))
        df = pd.read_csv(os.path.join(CSV_DIRECTORY, 'sp500_df.csv'), index_col='Symbol')

    symbols = df.index.tolist()
    calculate_features(symbols, df, FRED_KEY, crypto=crypto)

    df.dropna(inplace=True)
    features = df.columns.tolist()
    scaled_data = scale_data(df, features)
    apply_k_means(df, scaled_data, num_clusters=4)
    df.to_csv(os.path.join(CSV_DIRECTORY, 'processed_data.csv'))

    upload_to_gcs(GCS_BUCKET_NAME, os.path.join(CSV_DIRECTORY, 'processed_data.csv'), 'Data/processed_data.csv')

    send_email(SENDER_EMAIL, SENDER_EMAIL, 'ETL Process Complete', 'The ETL process has been successfully completed.', EMAIL_PASSWORD)

if __name__ == "__main__":
    main()
