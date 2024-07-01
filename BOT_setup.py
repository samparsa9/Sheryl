import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

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
    

def Create_list_of_tickers(dfindex):
    # Define the stock tickers
    tickers = []
    for ticker in dfindex:
        tickers.append(ticker)
    return tickers


def Calculate_features(tickers, df, batch_size=10):
    # Fetch historical data for SP500 
    sp500_data = yf.download('^GSPC', period='1y')
    sp500_data['Market Return'] = sp500_data['Adj Close'].pct_change()

    # Initialize columns for 200 SMA % Difference and beta value
    df['200 SMA % Difference'] = None
    df['beta value'] = None

    # Processing in batches
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        # Fetch data for this batch
        batch_data = yf.download(batch, period='1y', group_by='ticker')
        
        for ticker in batch:
            ticker_data = batch_data[ticker]
            ticker_data['Return'] = ticker_data['Adj Close'].pct_change()

            # Calculate 200-day SMA and percentage difference
            if len(ticker_data) >= 200:
                ticker_data['200 SMA'] = ticker_data['Adj Close'].rolling(window=200).mean()
                latest_close = ticker_data['Adj Close'].iloc[-1]
                latest_sma = ticker_data['200 SMA'].iloc[-1]
                if latest_sma != 0:  # Avoid division by zero
                    percent_diff = ((latest_close - latest_sma) / latest_sma) * 100
                    df.at[ticker, '200 SMA % Difference'] = percent_diff

            # Calculate beta value
            returns = pd.concat([ticker_data['Return'], sp500_data['Market Return']], axis=1).dropna()
            if len(returns) > 1:  # Ensure there are enough data points for covariance calculation
                covariance = np.cov(returns['Return'], returns['Market Return'])[0, 1]
                sp500_variance = np.var(returns['Market Return'])
                if sp500_variance != 0:  # Avoid division by zero
                    beta = covariance / sp500_variance
                    df.at[ticker, 'beta value'] = beta


def Scale_data(df):
    # Normalize data for clustering
    scaler = StandardScaler()
    # Changed line: Ensure only valid rows are scaled for clustering
    scaled_data = scaler.fit_transform(df[['200 SMA % Difference', 'beta value']].dropna())
    return scaled_data


def Apply_K_means(df, scaled_data, num_clusters=5):
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    # Initialize the Cluster column with NaN
    df['Cluster'] = np.nan
    # Assign clusters only to rows with non-null '200 SMA % Difference' and 'beta value'
    df.loc[df[['200 SMA % Difference', 'beta value']].dropna().index, 'Cluster'] = clusters


def Sort_and_save(df,file_path):
    # Sort DataFrame by Cluster
    df = df.sort_values(by='Cluster')

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the updated DataFrame to a CSV file
    df.to_csv(file_path, index=False)

def Save_cluster_df(df):
    df.to_csv('cluster_info.csv', index=True)


def plot_clusters(df):
    # Only plot rows with valid '200 SMA % Difference' and 'beta value'
    df = df.dropna(subset=['200 SMA % Difference', 'beta value', 'Cluster'])
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(df['200 SMA % Difference'], df['beta value'], c=df['Cluster'], cmap='viridis', alpha=0.6)
    
    # Adding color bar
    plt.colorbar(scatter, label='Cluster')
    
    plt.title('200 SMA % Difference vs Beta Value with Clusters')
    plt.xlabel('200 SMA % Difference')
    plt.ylabel('Beta Value')
    plt.grid(True)
    plt.show()

def cluster_df_setup(starting_cash, stock_df):
    portfolio_amt = starting_cash

    cluster_info_df = pd.DataFrame({
            "Cluster": [0,1,2,3,4],
            "Percentage": [0.2, 0.4, 0.2, 0.1, 0.1], #percentage of allocation for each cluster
            "Dollars In Cluster": [0] * 5,
            "Num Stocks": [0] * 5, #number of stocks in each cluster
            "Amount Per Stock": [0] * 5,
            "Tickers": [[], [], [], [], []] #list of tickers for each cluster
        })

    #set the portfolio amount for each cluster
    cluster_info_df["Dollars In Cluster"] = cluster_info_df["Percentage"] * portfolio_amt
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
        cluster_info_df.at[cluster, "Tickers"].append(ticker)
        cluster_info_df.at[cluster, "Num Stocks"] += 1

    for index, row in cluster_info_df.iterrows():
        # index is the cluster number
        if cluster_info_df.at[index, "Num Stocks"] > 0:
            cluster_info_df.at[index, "Amount Per Stock"] = cluster_info_df.at[index, "Dollars In Cluster"] / cluster_info_df.at[index, "Num Stocks"]
        else:
            cluster_info_df.at[index, "Amount Per Stock"] = 0  # Avoid division by zero
    
    return cluster_info_df



def main():
    location_of_sp500_csv_file = 'Sheryl/sp500_companies.csv'
    Create_sp500_csv(location_of_sp500_csv_file)
    stockdf = pd.read_csv(location_of_sp500_csv_file, index_col='Symbol')
    tickers = Create_list_of_tickers(stockdf.index)
    Calculate_features(tickers, stockdf)
    stockdf = stockdf.dropna()
    scaled_data = Scale_data(stockdf)
    Apply_K_means(stockdf, scaled_data)
    Sort_and_save(stockdf, location_of_sp500_csv_file)
    cluster_df = cluster_df_setup(1000000, stockdf)
    # Plot the clusters
    plot_clusters(stockdf)

if __name__ == "__main__":
    main()
