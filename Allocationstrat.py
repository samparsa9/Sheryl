import pandas as pd
import numpy as np
import yfinance as yf
import alpaca_trade_api as tradeapi
import time as tm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup

stockdf = pd.DataFrame({'Ticker': ['MMM', 'AXP', 'AMGN', 'AMZN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'DOW', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'V', 'WMT']})

stockdf.set_index('Ticker', inplace=True)


# tickerdf = pd.read_csv('dow_jones_companies.csv', index_col='Symbol')


# Initialize a column for the 200-day moving average percentage difference
stockdf['200 SMA % Difference'] = None

# Fetch historical data and calculate 200-day SMA and percentage difference for each symbol
for index, row in stockdf.iterrows():
    ticker = index  # Since 'Symbol' is now the index
    # Fetch historical data
    data = yf.download(ticker, period='1y')
    print(data)
    # Calculate the 200-day moving average
    if len(data) >= 200:
        data['200 SMA'] = data['Close'].rolling(window=200).mean()
        # Calculate the percentage difference
        latest_close = data['Close'].iloc[-1]
        latest_sma = data['200 SMA'].iloc[-1]
        if latest_sma != 0:  # Avoid division by zero
            percent_diff = ((latest_close - latest_sma) / latest_sma) * 100
            # Assign the percentage difference to the DataFrame
            stockdf.at[index, '200 SMA % Difference'] = percent_diff



# Save the updated DataFrame to a new CSV file (optional)
stockdf.to_csv('dow_jones_companies.csv', index=False)



# Display the updated DataFrame
print(stockdf)



# Normalize data
scaler = StandardScaler()
scaled_returns = scaler.fit_transform(stockdf)


# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(scaled_returns)
stockdf['Cluster'] = clusters

print(stockdf["Cluster"])

# url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"

# response = requests.get(url)
# soup = BeautifulSoup(response.text, 'html.parser')

# table = soup.find('table', {'id': 'constituents'})

# tickerdf = pd.read_html(str(table))[0]

# tickerdf.to_csv('dow_jones_companies.csv', index='Symbol')

