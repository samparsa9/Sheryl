import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def get_sp500_csv():
    # URL of the Wikipedia page
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Send a request to the webpage
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table containing the S&P 500 list
    table = soup.find('table', {'id': 'constituents'})

    # Read the table into a DataFrame
    df = pd.read_html(str(table))[0]

    # Display the first few rows of the DataFrame
    print(df.head())

    # # Save the DataFrame to a CSV file for reference
    df.to_csv('sp500_companies.csv', index=False)
    
#get_sp500_csv()

stockdf = pd.read_csv('sp500_companies.csv', index_col='Symbol')
# Define the stock tickers
tickers = []
for ticker in stockdf.index:
    tickers.append(ticker)

# Initialize DataFrame
#stockdf = pd.DataFrame(index=tickers)

# Fetch historical data for DJIA (Dow Jones Industrial Average)
sp500_data = yf.download('^SP500TR', period='1y')
print(sp500_data)
sp500_data['Market Return'] = sp500_data['Adj Close'].pct_change()

# Initialize columns for 200 SMA % Difference and beta value
stockdf['200 SMA % Difference'] = None
stockdf['beta value'] = None

# Fetch historical data and calculate 200 SMA and beta value for each stock
for ticker in tickers:
    ticker_data = yf.download(ticker, period='1y')
    ticker_data['Return'] = ticker_data['Adj Close'].pct_change()

    # Calculate 200-day SMA and percentage difference
    if len(ticker_data) >= 200:
        ticker_data['200 SMA'] = ticker_data['Adj Close'].rolling(window=200).mean()
        latest_close = ticker_data['Adj Close'].iloc[-1]
        latest_sma = ticker_data['200 SMA'].iloc[-1]
        if latest_sma != 0:  # Avoid division by zero
            percent_diff = ((latest_close - latest_sma) / latest_sma) * 100
            stockdf.at[ticker, '200 SMA % Difference'] = percent_diff

    # Calculate beta value
    returns = pd.concat([ticker_data['Return'], sp500_data['Market Return']], axis=1).dropna()
    if len(returns) > 1:  # Ensure there are enough data points for covariance calculation
        covariance = np.cov(returns['Return'], returns['Market Return'])[0, 1]
        sp500_variance = np.var(returns['Market Return'])
        if sp500_variance != 0:  # Avoid division by zero
            beta = covariance / sp500_variance
            stockdf.at[ticker, 'beta value'] = beta

# Normalize data for clustering
scaler = StandardScaler()
# Changed line: Ensure only valid rows are scaled for clustering
scaled_returns = scaler.fit_transform(stockdf[['200 SMA % Difference', 'beta value']].dropna())

# Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(scaled_returns)
# Initialize the Cluster column with NaN
stockdf['Cluster'] = np.nan
# Assign clusters only to rows with non-null '200 SMA % Difference' and 'beta value'
stockdf.loc[stockdf[['200 SMA % Difference', 'beta value']].dropna().index, 'Cluster'] = clusters

# Sort DataFrame by Cluster
stockdf = stockdf.sort_values(by='Cluster')

# Save the updated DataFrame to a CSV file
stockdf.to_csv('sp500_companies.csv', index=True)

# Display the updated DataFrame
#print(stockdf)

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

# Plot the clusters
plot_clusters(stockdf)
