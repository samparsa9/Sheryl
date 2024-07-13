import os
import pandas as pd
import sys
# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import necessary configuration, data collection, processing, and visualization modules
from config.settings import CSV_DIRECTORY, FRED_KEY
from src.data_collection.sp500_scraper import create_sp500_csv
from src.data_collection.crypto_scraper import create_crypto_csv
from src.data_collection.calculate_features import calculate_features
from src.data_processing.scaler import scale_data
from src.data_processing.k_means_cluster import apply_k_means
from src.utils.alpaca_utils import cluster_and_allocation_setup
from src.data_visualization.plotting import plot_features, plot_features_1, plot_features_2, plot_features_interactive

def main():
    # Ensure the CSV directory exists, create it if it does not
    os.makedirs(CSV_DIRECTORY, exist_ok=True)
    # Flag to determine whether to process cryptocurrency data or S&P 500 data
    crypto = False

    if crypto:
        # Create a CSV file containing cryptocurrency data
        create_crypto_csv(os.path.join(CSV_DIRECTORY, 'crypto_df.csv'))
        # Load the cryptocurrency data from the created CSV file into a DataFrame
        df = pd.read_csv(os.path.join(CSV_DIRECTORY, 'crypto_df.csv'), index_col='Symbol')
    else:
        # Create a CSV file containing S&P 500 data
        create_sp500_csv(os.path.join(CSV_DIRECTORY, 'sp500_df.csv'))
        # Load the S&P 500 data from the created CSV file into a DataFrame
        df = pd.read_csv(os.path.join(CSV_DIRECTORY, 'sp500_df.csv'), index_col='Symbol')

    # Extract the list of symbols (index) from the DataFrame
    symbols = df.index.tolist()
    # Calculate additional features for the data
    calculate_features(symbols, df, FRED_KEY, crypto=crypto)
    # Drop rows with any missing values
    df.dropna(inplace=True)
    # Extract the list of feature names from the DataFrame columns
    features = df.columns.tolist()
    # Scale the data using the specified features
    scaled_data = scale_data(df, features)
    # Apply K-Means clustering to the scaled data and update the DataFrame with cluster assignments
    apply_k_means(df, scaled_data, num_clusters=4)
    # Sort the DataFrame based on cluster assignments
    df = df.sort_values(by='Cluster')
    # Save the processed DataFrame to a new CSV file
    df.to_csv(os.path.join(CSV_DIRECTORY, 'processed_data.csv'))

    # Set up clustering and allocation with a specified initial capital
    new_df = cluster_and_allocation_setup(1000, df, 4, True)

    print(new_df)
    # Generate interactive plots for the processed data
    plot_features_interactive(df)


    # send_email(SENDER_EMAIL, SENDER_EMAIL, 'ETL Process Complete', 'The ETL process has been successfully completed.', EMAIL_PASSWORD)

if __name__ == "__main__":
    main()
