import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import Alpacahelperfuncs as hf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import alpaca_trade_api as tradeapi
import random
from fredapi import Fred
import certifi
import ssl
import urllib.request
import time as tm
import seaborn as sns
load_dotenv()
# Email Feature info
sender = os.getenv('sender')
recipient = os.getenv('sender')
password = os.getenv('email_password')

# Alpaca Info
api_key = os.getenv('api_key')
api_secret = os.getenv("api_secret")
base_url = os.getenv('base_url')

#fred info
# Ensure the urllib uses the certifi certificate bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Set up FRED API key (you need to sign up at https://fred.stlouisfed.org/ to get an API key)
fred_key = os.getenv('fred_key')
fred = Fred(api_key=fred_key)



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


def Create_sp500_csv(file_path):
    # URL of the Wikipedia page
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Send a request to the webpage
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table containing the S&P 500 list
    table = soup.find('table', {'id': 'constituents'})

    # Read the table into a DataFrame
    df = pd.read_html(str(table))[0]

    # Getting rid of unnecessary columns and dropping any rows that might have NaN values
    df = df.drop(columns=["Security", "GICS Sector", "GICS Sub-Industry", "Headquarters Location", "Date added", "CIK", "Founded"])
    df = df.dropna()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # # Save the DataFrame to a CSV file for reference
    df.to_csv(file_path, index=False)

def create_crypto_csv(file_path):

    crypto_df = pd.DataFrame({
            "Symbol": ["AAVE/USD","AVAX/USD","BAT/USD","BCH/USD","BTC/USD","CRV/USD","DOGE/USD","DOT/USD","ETH/USD","LINK/USD","LTC/USD","MKR/USD","SHIB/USD","SUSHI/USD","UNI/USD","USDC/USD","USDT/USD","XTZ/USD"]
        })# "GRT/USD"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the DataFrame to a CSV file
    crypto_df.to_csv(file_path, index=False)


def Create_list_of_tickers(dfindex):
    # Define the stock tickers
    tickers = []
    for ticker in dfindex:
        tickers.append(ticker)
    return tickers

def get_list_of_features(df):
    return list(df.columns)

def Calculate_features(symbols, df, batch_size=10, crypto=False):
    
    # Fetch historical data for Bitcoin
    BTC_data = yf.download('BTC-USD', period='1y')    
    BTC_data['Market Return'] = BTC_data['Adj Close'].pct_change()

    # getting treasury data for Sharpe Ratio
    series_id = 'GS1'
    treasury_data = fred.get_series(series_id)
    # Get the latest risk-free rate
    if not treasury_data.empty:
        risk_free_rate = treasury_data.iloc[-1] / 100  # Converting to decimal form
        # print(f"Risk-Free Rate (1-year U.S. Treasury): {risk_free_rate:.4f}")
    else:
        print("1-year U.S. Treasury yield data is not available.")

    # Processing in batches
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        # Replace "-" with "/" in each string in the batch list so yfinance can fetch data
        if crypto:
            batch = [symbol.replace("/", "-") for symbol in batch]

        # Fetch data for this batch
        batch_data = yf.download(batch, period='1y', group_by='ticker')

        for ticker in batch:
            ticker_data = batch_data[ticker]
            if isinstance(ticker_data, pd.DataFrame):
                ticker_data = ticker_data.copy()  # Make a copy to avoid the warning

                # Calculate returns
                ticker_data['Return'] = ticker_data['Adj Close'].pct_change()
                # Adding volume column
                df.at[ticker, "Daily $ Volume"] = ticker_data.at[(list(ticker_data.index)[-1]), 'Volume']

                # Calculate 200-day SMA percentage difference, 50-day SMA percentage difference
                if len(ticker_data) >= 50:
                    ticker_data['50 SMA'] = ticker_data['Adj Close'].rolling(window=50).mean()
                    latest_close = ticker_data['Adj Close'].iloc[-1]
                    # print(f"Latest close for {ticker} is: {latest_close}")
                    latest_50_sma = ticker_data['50 SMA'].iloc[-1]
                    # print(f"Latest 50 sma for {ticker} is: {latest_50_sma}")
                    if latest_50_sma != 0:
                        percent_diff_50_sma = ((latest_close - latest_50_sma) / latest_50_sma) * 100
                        df.at[ticker, '50 SMA % Difference'] = percent_diff_50_sma

                if len(ticker_data) >= 200:
                    ticker_data['200 SMA'] = ticker_data['Adj Close'].rolling(window=200).mean()
                    latest_200_sma = ticker_data['200 SMA'].iloc[-1]
                    if latest_200_sma != 0:
                        percent_diff_200_sma = ((latest_close - latest_200_sma) / latest_200_sma) * 100
                        df.at[ticker, '200 SMA % Difference'] = percent_diff_200_sma

                # Calculate 50-day and 200-day EMA percentage difference
                if len(ticker_data) >= 50:
                    ticker_data['50 EMA'] = ticker_data['Adj Close'].ewm(span=50, adjust=False).mean()
                    latest_50_ema = ticker_data['50 EMA'].iloc[-1]
                    if latest_50_ema != 0:
                        percent_diff_50_ema = ((latest_close - latest_50_ema) / latest_50_ema) * 100
                        df.at[ticker, '50 Day EMA % Difference'] = percent_diff_50_ema

                if len(ticker_data) >= 200:
                    ticker_data['200 EMA'] = ticker_data['Adj Close'].ewm(span=200, adjust=False).mean()
                    latest_200_ema = ticker_data['200 EMA'].iloc[-1]
                    if latest_200_ema != 0:
                        percent_diff_200_ema = ((latest_close - latest_200_ema) / latest_200_ema) * 100
                        df.at[ticker, '200 Day EMA % Difference'] = percent_diff_200_ema

                # Calculate annualized beta value with respect to btcusd as market return
                returns = pd.concat([ticker_data['Return'], BTC_data['Market Return']], axis=1).dropna()
                if len(returns) > 1:  # Ensure there are enough data points for covariance calculation
                    covariance = np.cov(returns['Return'], returns['Market Return'])[0, 1]
                    BTC_variance = np.var(returns['Market Return'])
                    if BTC_variance != 0:  # Avoid division by zero
                        beta = covariance / BTC_variance
                        df.at[ticker, 'Beta value'] = beta

                # Calculate Annualized Sharpe Ratio
                daily_risk_free_rate = risk_free_rate / 252
                ticker_data['Excess Return'] = ticker_data['Return'] - daily_risk_free_rate

                mean_excess_return = ticker_data['Excess Return'].mean()
                std_excess_return = ticker_data['Excess Return'].std()

                if std_excess_return != 0:
                    sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(252)
                    df.at[ticker, 'Sharpe Ratio'] = sharpe_ratio
                    
                # Volatility
                df.at[ticker, 'Volatility'] = ticker_data['Return'].std() * np.sqrt(252)

                # Calculating NVT Ratio: (market cap of coin / daily volume)
                # if crypto:
                #     # symbol = ticker.replace("-USD", "").lower()
                #     df.at[ticker, 'Market Cap'] = get_crypto_market_cap(ticker)
                #     tm.sleep(1)
                #     df.at[ticker, 'Trading Volume in $'] = ticker_data['Volume'].mean()
                #     NVT = df.at[ticker, 'Market Cap'] / df.at[ticker, 'Trading Volume in $']

            
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
            market_value = hf.get_market_value(api, ticker, crypto)
            # print(f"Ticker: {ticker}, Market Value: {market_value}")
            dollars_in_this_cluster += market_value
        # Populating the Dollars In Cluster column with these new values in our current portfolio allocation df
        current_portfolio_allocation.loc[cluster, "Dollars In Cluster"] = round(dollars_in_this_cluster, 2)
        # print(f"Cluster {cluster}, Dollars In Cluster: {dollars_in_this_cluster}")
    for cluster in optimal_portfolio_allocation_df.index:
        dollars_in_this_cluster = current_portfolio_allocation.loc[cluster, 'Dollars In Cluster']
        account_value = hf.get_total_account_value(api)

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

def Is_balanced(current_portfolio_df, api):
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

        hf.execute_trade('sell', amount, ticker, api, notional=True, crypto=crypto)


    # Randomly select tickers to buy
    if available_tickers is None:
        print("No available tickers to buy from.")
        return

    tickers_to_buy = random.sample(available_tickers, buy_count)
    for ticker in tickers_to_buy:
        ticker = ticker.replace("-", "")  # Remove special characters if needed
        ticker = ticker.replace("/", "")
        amount = 30  # Define a fixed amount to buy or calculate based on your logic

        hf.execute_trade('buy', amount, ticker, api, notional=True, crypto=crypto)


def main():
    # Initialize Alpaca API
    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
    # Location of our main info dataframe
    location_of_main_csv_file = 'Sheryl/crypto_data.csv' #Change this to change what symbol we are looking at

    # Location of our cluster info dataframe
    location_of_cluster_csv_file = 'Sheryl/symbol_cluster_info.csv'

    # SET BACK TO 5
    num_clusters = 4 # Use this to specify how many cluster for K-means

    # Set true since were dealing with crypto symbols
    crypto = True # SET BACK TO FALSE

    # Running wikapedia scraper to populate initial sp500 csv with ticker symbols of sp500 companies
    create_crypto_csv(location_of_main_csv_file) #CHANGE THIS BACK FOR SP500

    # Setting our stockdf to be the csv file we just created
    og_df = pd.read_csv(location_of_main_csv_file, index_col='Symbol')

    # Creating our tickers list which we will pass into the Calculate_features function
    tickers = Create_list_of_tickers(og_df.index)

    # Calculating our features for each ticker and populating the dataframe with these values
    Calculate_features(tickers, og_df, crypto=crypto)
    print("Before dropping na")
    print(og_df)
    # Dropping any columns that may have resulted in NaN values
    og_df = og_df.dropna()

    print("After dropping")
    print(og_df)
    # Creating a scaled_data numpy array that we will pass into our K means algorithm
    scaled_data = Scale_data(og_df)

    # Running k means
    Apply_K_means(og_df, scaled_data, num_clusters)

    # Soring the data frame based on cluster value and saving these values to the dataframe, and then updating the csv
    Sort_and_save(og_df, location_of_main_csv_file)

    # Creating a new dataframe that will contain information about the clusters that will be used for trading logic
    optimal_portfolio_allocation_info_df = cluster_df_setup(hf.get_total_account_value(api), og_df)

    print("---------------------OPTIMAL PORTFOLIO ALLOCATION BASED ON TOTAL CURRENT ACCOUNT VALUE---------------------") 
    print(optimal_portfolio_allocation_info_df)
    print("-----------------------------------------------------------------------------------------------------------")

    # Save the newsly created cluster df to a csv file
    Sort_and_save(optimal_portfolio_allocation_info_df, location_of_cluster_csv_file)
    
    print(get_list_of_features(og_df))
    # Plotting cluster data
    plot_features(og_df)
    pass
if __name__ == "__main__":
    main()
