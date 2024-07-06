import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import src.utils.alpaca_utils as hf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import random
import seaborn as sns
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
import data_collection as dc
from google.cloud import storage

load_dotenv()
# Email Feature info
sender = os.getenv('sender')
recipient = os.getenv('sender')
password = os.getenv('email_password')

# Alpaca Info
api_key = os.getenv('api_key')
api_secret = os.getenv("api_secret")
base_url = os.getenv('base_url')

# Data directory
csv_directory = os.getenv('DATA_directory')
if not csv_directory:
    raise ValueError("CSV_DIRECTORY environment variable not set")
# Ensure the directory exists
os.makedirs(csv_directory, exist_ok=True)

def send_email(subject, message):

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject

    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        text = msg.as_string()
        server.sendmail(sender, recipient, text)
        server.quit()
        # print('Email sent')
    except Exception as e:
        print(f"Failed to send email: {e}")


def Create_list_of_tickers(dfindex):
    # Define the stock tickers
    tickers = []
    for ticker in dfindex:
        tickers.append(ticker)
    return tickers

def get_list_of_features(df):
    return list(df.columns)


            
# def get_crypto_market_cap(symbol):
#     # Map some common cryptocurrency symbols to CoinGecko IDs
#     symbol_map = {
#         'BTC-USD': 'bitcoin',
#         'ETH-USD': 'ethereum',
#         'AAVE-USD': 'aave',
#         'AVAX-USD': 'avalanche-2',
#         'BAT-USD': 'basic-attention-token',
#         'BCH-USD': 'bitcoin-cash',
#         'CRV-USD': 'curve-dao-token',
#         'DOGE-USD': 'dogecoin',
#         'DOT-USD': 'polkadot',
#         'LINK-USD': 'chainlink',
#         'LTC-USD': 'litecoin',
#         'MKR-USD': 'maker',
#         'SHIB-USD': 'shiba-inu',
#         'SUSHI-USD': 'sushi',
#         'UNI-USD': 'uniswap',
#         'USDC-USD': 'usd-coin',
#         'USDT-USD': 'tether',
#         'XTZ-USD': 'tezos'
#     }
    
#     # Convert to lowercase and look up in the symbol_map
#     symbol_id = symbol_map.get(symbol.upper())
#     if not symbol_id:
#         print(f"Symbol {symbol} not found in the symbol map.")
#         return None

#     url = f'https://api.coingecko.com/api/v3/coins/{symbol_id}'
#     retries = 5
#     for i in range(retries):
#         try:
#             response = requests.get(url)
#             if response.status_code == 200:
#                 data = response.json()
#                 return data['market_data']['market_cap']['usd']
#             else:
#                 print(f"Failed to fetch market cap for {symbol_id}. HTTP Status code: {response.status_code}")
#                 if response.status_code == 429:
#                     # If rate limited, wait before retrying
#                     tm.sleep(2 ** i)
#         except Exception as e:
#             print(f"Error fetching data for {symbol_id}: {e}")
#             tm.sleep(2 ** i)  # Exponential backoff in case of other errors
        
#         return None

def Scale_data(df):
    # Normalize data for clustering
    scaler = StandardScaler()
    # Changed line: Ensure only valid rows are scaled for clustering
    scaled_data = scaler.fit_transform(df[get_list_of_features(df)].dropna())
    return scaled_data


def Apply_K_means(df, scaled_data, num_clusters=5):
    feature_list = get_list_of_features(df)
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    # Initialize the Cluster column with NaN
    df['Cluster'] = np.nan
    # Assign clusters only to rows with non-null
    valid_rows = df[feature_list].dropna().index
    df.loc[valid_rows, 'Cluster'] = clusters


def Sort_and_save(df,file_path):
    # Sort DataFrame by Cluster
    df = df.sort_values(by='Cluster')

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the updated DataFrame to a CSV file
    df.to_csv(file_path, index=True)


def plot_features(df):
    # Check if 'Cluster' column exists in the DataFrame
    if 'Cluster' not in df.columns:
        raise ValueError("DataFrame must contain a 'Cluster' column")
    
    # Standardize the features
    scaler = StandardScaler()
    features = df.columns.drop('Cluster')
    scaled_features = scaler.fit_transform(df[features])
    
    # List of classifiers for outlier detection
    classifiers = {
        'KNN': KNN(),
        'Isolation Forest': IForest(),
        'LOF': LOF()
    }
    
    # Select feature pairs to plot
    feature_pairs = [
        ('Daily $ Volume', '50 SMA % Difference'),
        ('200 SMA % Difference', '50 Day EMA % Difference'),
        ('200 Day EMA % Difference', 'Beta value'),
        ('Sharpe Ratio', 'Volatility')
    ]
    
    # Initialize the plot
    num_classifiers = len(classifiers)
    fig, axes = plt.subplots(len(feature_pairs), num_classifiers, figsize=(15, 5 * len(feature_pairs)))
    
    if len(feature_pairs) == 1:
        axes = [axes]
    
    for row, (feature_x, feature_y) in enumerate(feature_pairs):
        # Select the current pair of features
        X = df[[feature_x, feature_y]].values
        X_scaled = scaler.fit_transform(X)
        
        for col, (clf_name, clf) in enumerate(classifiers.items()):
            # Fit the model
            clf.fit(X_scaled)
            
            # Predict the results
            y_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
            y_scores = clf.decision_scores_  # raw outlier scores
            
            # Create a grid for contour plot
            xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 200),
                                 np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 200))
            zz = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            zz = zz.reshape(xx.shape)
            
            # Scatter plot with cluster and outlier information
            ax = axes[row][col]
            scatter = ax.scatter(X[:, 0], X[:, 1], c=df['Cluster'], cmap='viridis', label='Clusters', alpha=0.6, edgecolor='k')
            outliers = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', marker='x', label='Outliers', alpha=0.8)
            
            # Contour plot
            ax.contourf(xx, yy, zz, levels=np.linspace(zz.min(), zz.max(), 10), cmap='coolwarm', alpha=0.3)
            ax.contour(xx, yy, zz, levels=[0], linewidths=2, colors='red')
            
            # Density plot
            sns.kdeplot(x=X[:, 0], y=X[:, 1], ax=ax, fill=True, cmap="Blues", alpha=0.1)
            
            # Title and legend
            ax.set_title(f"{clf_name} Outlier Detection\nFeatures: {feature_x} vs {feature_y}")
            if col == 0:
                ax.set_ylabel(f"{feature_y}")
            if row == len(feature_pairs) - 1:
                ax.set_xlabel(f"{feature_x}")
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)
            legend2 = ax.legend(['Inliers', 'Outliers'], loc='upper right')
            ax.add_artist(legend2)
    
    # Show plot
    plt.suptitle("Outlier Detection with Different Algorithms", size=30)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_features_2(df):
    # Check if 'Cluster' column exists in the DataFrame
    if 'Cluster' not in df.columns:
        raise ValueError("DataFrame must contain a 'Cluster' column")
    
    # Standardize the features
    scaler = StandardScaler()
    features = df.columns.drop('Cluster')
    scaled_features = scaler.fit_transform(df[features])
    
    # List of classifiers for outlier detection
    classifiers = {
        'KNN': KNN(),
        'Isolation Forest': IForest(),
        'LOF': LOF()
    }
    
    # Initialize the plot
    num_classifiers = len(classifiers)
    fig, axes = plt.subplots(num_classifiers, figsize=(15, 5 * num_classifiers))
    
    if num_classifiers == 1:
        axes = [axes]
    
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        # Fit the model
        clf.fit(scaled_features)
        
        # Predict the results
        y_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        
        # Scatter plot with cluster and outlier information
        ax = axes[i]
        scatter = ax.scatter(df[features[0]], df[features[1]], c=df['Cluster'], cmap='viridis', label='Clusters', alpha=0.6)
        outliers = ax.scatter(df[features[0]], df[features[1]], c=y_pred, cmap='coolwarm', marker='x', label='Outliers', alpha=0.8)
        
        # Density plot
        sns.kdeplot(x=df[features[0]], y=df[features[1]], ax=ax, fill=True, cmap="Blues", alpha=0.3)
        
        # Title and legend
        ax.set_title(f"{clf_name} Outlier Detection")
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        legend2 = ax.legend(['Inliers', 'Outliers'], loc='upper right')
        ax.add_artist(legend2)
    
    # Show plot
    plt.suptitle("Outlier Detection with Different Algorithms", size=30)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_features_1(df):
    # Check if 'Cluster' column exists in the DataFrame
    if 'Cluster' not in df.columns:
        raise ValueError("DataFrame must contain a 'Cluster' column")
    
    # Get the list of features excluding the 'Cluster' column
    features = df.columns.drop('Cluster')
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of Features")
    plt.show()

    # Plot pair plot for selected features
    selected_features = ['Daily $ Volume', '50 SMA % Difference', '200 SMA % Difference', '50 Day EMA % Difference']  # Select top features
    sns.pairplot(df, vars=selected_features, hue="Cluster", palette="husl", markers=["o", "s", "D", "P", "X"])
    plt.suptitle("Pair Plot of Selected Features by Cluster", y=1.02)
    plt.show()

    # Display descriptive statistics
    print("Descriptive Statistics of Features:")
    print(df[features].describe().transpose())


def cluster_df_setup(starting_cash, stock_df):
    portfolio_amt = starting_cash

    # CHANGE ALL THIS BACK
    cluster_info_df = pd.DataFrame({
             #"Cluster": [0,1,2,3,4],
            "Cluster": [0,1,2,3],
            #"Percentage": [0.2, 0.4, 0.2, 0.1, 0.1], #percentage of allocation for each cluster
            "Percentage": [0.2, 0.4, 0.2, 0.2],
            #"Dollars In Cluster": [0.0] * 5.0,
            "Dollars In Cluster": [0.0] * 4,
            #"Num Stocks": [0.0] * 5, #number of stocks in each cluster
            "Num Stocks": [0.0] * 4,
            #"Amount Per Stock": [0.0] * 5,
            "Amount Per Stock": [0.0] * 4,
            #"Tickers": [[], [], [], [], []] #list of tickers for each cluster
            "Tickers": [[], [], [], []]
        })

    # Set the portfolio amount for each cluster
    cluster_info_df["Dollars In Cluster"] = round((cluster_info_df["Percentage"] * portfolio_amt),2)
    cluster_info_df.set_index("Cluster", inplace=True)
    #figure out how many stocks are in each cluster, and fill tickers column
    for ticker, row in stock_df.iterrows():
        # print("----------------------------------------------------------------------------")
        # print(ticker)
        # print(row)
        # print("----------------------------------------------------------------------------")
        cluster = row["Cluster"]
        # print(stock_df.head())
        # print(cluster_info_df)
        cluster_info_df.loc[cluster, "Tickers"].append(ticker)
        cluster_info_df.loc[cluster, "Num Stocks"] += 1

    for index, row in cluster_info_df.iterrows():
        # index is the cluster number
        if cluster_info_df.loc[index, "Num Stocks"] > 0:
            cluster_info_df.loc[index, "Amount Per Stock"] = round((cluster_info_df.loc[index, "Dollars In Cluster"] / cluster_info_df.loc[index, "Num Stocks"]), 2)
        else:
            cluster_info_df.loc[index, "Amount Per Stock"] = 0  # Avoid division by zero
    
    return cluster_info_df

def Get_current_portfolio_allocation(optimal_portfolio_allocation_df, api, crypto=False):
    # Create a new dataframe that represents our current portfolios dollar allocation each clusters
    current_portfolio_allocation = pd.DataFrame({
            #"Cluster": [0,1,2,3,4],
            "Cluster": [0,1,2,3],
            "Current Pct Allocation": [0.0] * 4,
            "Pct Off From Optimal": [0.0] * 4,
            #"Dollars In Cluster": [0] * 5,
            "Dollars In Cluster": [0.0] * 4
        })
    current_portfolio_allocation.set_index("Cluster", inplace=True)
    # This for loop will be used to calulate the total dollars in each cluster by looping through each ticker in each cluster
    # Snd adding the market value of our position in that ticker to a running sum
    for cluster in optimal_portfolio_allocation_df.index:
        dollars_in_this_cluster = 0.0
        tickers = optimal_portfolio_allocation_df.loc[cluster, "Tickers"]
        for ticker in tickers:
            #print("here is the ticker before its sent to the func: " + ticker)
            market_value = hf.get_market_value(ticker, crypto)
            # print(f"Ticker: {ticker}, Market Value: {market_value}")
            dollars_in_this_cluster += market_value
        # Populating the Dollars In Cluster column with these new values in our current portfolio allocation df
        current_portfolio_allocation.loc[cluster, "Dollars In Cluster"] = round(dollars_in_this_cluster, 2)
        # print(f"Cluster {cluster}, Dollars In Cluster: {dollars_in_this_cluster}")
    for cluster in optimal_portfolio_allocation_df.index:
        dollars_in_this_cluster = current_portfolio_allocation.loc[cluster, 'Dollars In Cluster']
        account_value = hf.get_total_account_value()

        current_cluster_pct_allocation = (dollars_in_this_cluster / account_value)
        optimal_cluster_pct_allocation = optimal_portfolio_allocation_df.loc[cluster, "Percentage"]

        current_portfolio_allocation.loc[cluster, "Current Pct Allocation"] = current_cluster_pct_allocation
        current_portfolio_allocation.loc[cluster, "Pct Off From Optimal"] = current_cluster_pct_allocation - optimal_cluster_pct_allocation

    return current_portfolio_allocation


def Get_most_unoptimized_cluster(current_portfolio_df):
    most_unoptimized_cluster = None
    largest_pct_off_by = 0
    for cluster, row in current_portfolio_df.iterrows():
        this_cluster_off_by = current_portfolio_df.loc[cluster, "Pct Off From Optimal"]
        if abs(this_cluster_off_by) > abs(largest_pct_off_by):
            largest_pct_off_by = this_cluster_off_by
            most_unoptimized_cluster = cluster
    return most_unoptimized_cluster, largest_pct_off_by

def Get_most_optimized_cluster(current_portfolio_df):
    most_optimized_cluster = None
    lowest_pct_off_by = 1
    for cluster, row in current_portfolio_df.iterrows():
        this_cluster_off_by = current_portfolio_df.loc[cluster, "Pct Off From Optimal"]
        if abs(this_cluster_off_by) < abs(lowest_pct_off_by):
            lowest_pct_off_by = this_cluster_off_by
            most_optimized_cluster = cluster
    return most_optimized_cluster, lowest_pct_off_by


def Get_most_unoptimized_clusters(optimal_portfolio_allocation_df, current_portfolio_allocation, api):
    """
    This function will take in the cluster information dataframe and the api,
    it will then see if the portfolio is balanced correctly depending on the cluster allocations,
    if it is not, it will return the two clusters off by the most percentage
    """
    # Initializing variables to represent the highest unoptimal cluster and how much higher than optimal it is
    # And the lowest unoptimal cluster and how much lower than optimal it is
    Highest_unoptimal_allocation_pct = -float('inf')
    Lowest_unoptimal_allocation_pct = float('inf')
    Highest_unoptimal_allocation_cluster = 0
    Lowest_unoptimal_allocation_cluster = 0

    # For every cluster, compare current to optimal percent
    for index, row in optimal_portfolio_allocation_df.iterrows():
        # Retreiving and storing this clusters current dollar allocation
        current_pct_allocation = current_portfolio_allocation.loc[index, "Current Pct Allocation"]
        # Retreiving and storing this clusters optimal dollar allocation
        optimal_pct_allocation = optimal_portfolio_allocation_df.loc[index, "Percentage"]


        # Storing the % diff higher or lower than optimal the current allocation is
        pct_diff_between_current_and_optimal_allocation = current_pct_allocation - optimal_pct_allocation
        # print(f'pct_diff_between_current_and_optimal_allocation: {pct_diff_between_current_and_optimal_allocation}\ncurrent_dollar_allocation: {current_dollar_allocation}\noptimal_dollar_allocation: {optimal_dollar_allocation}')
        # If the % diff is higher than the current highest unoptimal allocation, set it to this new % diff and update which cluster
        if pct_diff_between_current_and_optimal_allocation > Highest_unoptimal_allocation_pct:
            Highest_unoptimal_allocation_pct = pct_diff_between_current_and_optimal_allocation
            Highest_unoptimal_allocation_cluster = index
        # If the % diff is lower than the current lowest unoptimal allocation, set it to this new % diff and update which cluster
        elif pct_diff_between_current_and_optimal_allocation < Lowest_unoptimal_allocation_pct:
            Lowest_unoptimal_allocation_pct = pct_diff_between_current_and_optimal_allocation
            Lowest_unoptimal_allocation_cluster = index

    # Returning the values as a tuple of tuples
    tuple_to_return = ((Highest_unoptimal_allocation_cluster, Highest_unoptimal_allocation_pct), (Lowest_unoptimal_allocation_cluster, Lowest_unoptimal_allocation_pct))
    return tuple_to_return

def Is_balanced(current_portfolio_df):
    largest_pct_off_by = 0
    for cluster, row in current_portfolio_df.iterrows():
        this_cluster_off_by = current_portfolio_df.loc[cluster, "Pct Off From Optimal"]
        if abs(this_cluster_off_by) > abs(largest_pct_off_by):
            largest_pct_off_by = this_cluster_off_by
    return abs(largest_pct_off_by) < 0.03 #changed threshold

def calculate_seconds_till_next_reallocation(timezone, hour_to_trade, minute_to_trade):
                now = datetime.now(timezone)
                target_time = now.replace(hour=hour_to_trade, minute=minute_to_trade, second=0, microsecond=0)
                if now > target_time:
                    target_time += timedelta(days=1)
                return int((target_time - now).total_seconds())

def throw_off_portfolio(api, sell_count=3, buy_count=3, available_tickers=None, crypto=False):
    """
    Randomly sells a number of tickers from the current portfolio and buys a number of new random tickers.

    :param api: Alpaca API instance
    :param sell_count: Number of tickers to sell
    :param buy_count: Number of tickers to buy
    :param available_tickers: List of available tickers to buy from
    :param crypto: Boolean indicating if the tickers are cryptocurrencies
    """
    # Fetch current positions
    positions = api.list_positions()
    if len(positions) == 0:
        print("No positions to sell.")
        return

    # Randomly select tickers to sell
    tickers_to_sell = random.sample(positions, min(sell_count, len(positions)))
    for position in tickers_to_sell:
        ticker = position.symbol
        ticker = ticker.replace("-", "")  # Remove special characters if needed
        ticker = ticker.replace("/", "")
        amount = 30

        hf.execute_trade('sell', amount, ticker, notional=True, crypto=crypto)


    # Randomly select tickers to buy
    if available_tickers is None:
        print("No available tickers to buy from.")
        return

    tickers_to_buy = random.sample(available_tickers, buy_count)
    for ticker in tickers_to_buy:
        ticker = ticker.replace("-", "")  # Remove special characters if needed
        ticker = ticker.replace("/", "")
        amount = 30  # Define a fixed amount to buy or calculate based on your logic

        hf.execute_trade('buy', amount, ticker, notional=True, crypto=crypto)

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    # Path to your service account key file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'Sheryl/teak-environs-428403-r1-13b20cb353f4.json'

    # Initialize a storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Create a blob (object) in the bucket
    blob = bucket.blob(destination_blob_name)

    # Upload the file to the blob
    blob.upload_from_filename(source_file_name)


def main():
    # Initialize Alpaca API
    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
    # Location of our main info dataframe

    # SET BACK TO 5
    num_clusters = 4 # Use this to specify how many cluster for K-means

    # Set true since were dealing with crypto symbols
    crypto = True # SET BACK TO FALSE

    # Running wikapedia scraper to populate initial sp500 csv with ticker symbols of sp500 companies
    # Running wikapedia scraper to populate initial sp500 csv with ticker symbols of sp500 companies
    dc.create_crypto_csv() #CHANGE THIS BACK FOR SP500

    # Setting our stockdf to be the csv file we just created
    og_df = pd.read_csv(csv_directory + 'crypto_df.csv', index_col='Symbol')

    # Creating our tickers list which we will pass into the Calculate_features function
    tickers = Create_list_of_tickers(og_df.index)

    # Calculating our features for each ticker and populating the dataframe with these values
    dc.Calculate_features(tickers, og_df, crypto=crypto)

    # Dropping any columns that may have resulted in NaN values
    og_df = og_df.dropna()

    # Creating a scaled_data numpy array that we will pass into our K means algorithm
    scaled_data = Scale_data(og_df)

    # Running k means
    Apply_K_means(og_df, scaled_data, num_clusters)

    # Soring the data frame based on cluster value and saving these values to the dataframe, and then updating the csv
    Sort_and_save(og_df, csv_directory + 'crypto_df.csv')

    # Upload the CSV to Google Cloud Storage
    bucket_name = 'sherylgcsdatabucket'
    destination_blob_name = 'Data/crypto_df.csv'
    upload_to_gcs(bucket_name, csv_directory + 'crypto_df.csv', destination_blob_name)

    # Creating a new dataframe that will contain information about the clusters that will be used for trading logic
    optimal_portfolio_allocation_info_df = cluster_df_setup(hf.get_total_account_value(), og_df)

    print("---------------------OPTIMAL PORTFOLIO ALLOCATION BASED ON TOTAL CURRENT ACCOUNT VALUE---------------------") 
    print(optimal_portfolio_allocation_info_df)
    print("-----------------------------------------------------------------------------------------------------------")

    # Save the newsly created cluster df to a csv file
    Sort_and_save(optimal_portfolio_allocation_info_df, csv_directory + 'symbol_cluster_info.csv')
    destination_blob_name = 'Data/symbol_cluster_info.csv'
    upload_to_gcs(bucket_name, csv_directory + 'symbol_cluster_info.csv', destination_blob_name)
    
    # print(get_list_of_features(og_df))
    # Plotting cluster data
    # plot_features(og_df)
    pass
if __name__ == "__main__":
    main()