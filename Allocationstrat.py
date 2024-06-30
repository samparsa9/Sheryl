import pandas as pd
import numpy as np
import yfinance as yf
import alpaca_trade_api as tradeapi
import time as tm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup

def fetch_data(tickers, start_date='2020-01-01', end_date='2023-01-01'):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.dropna()
    return returns

tickerdf = pd.read_csv('Sheryl/dow_jones_companies.csv')
ticker_list = []
for ticker in tickerdf['Symbol']:
    ticker_list.append(ticker)

print(ticker_list)

data = fetch_data(ticker_list)

print(data)


# url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"

# response = requests.get(url)
# soup = BeautifulSoup(response.text, 'html.parser')

# table = soup.find('table', {'id': 'constituents'})

# tickerdf = pd.read_html(str(table))[0]

# tickerdf.to_csv('dow_jones_companies.csv', index='Symbol')

