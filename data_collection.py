import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import os
from fredapi import Fred
import certifi
import ssl


#CSV location
csv_directory = os.getenv('DATA_directory')
if not csv_directory:
    raise ValueError("CSV_DIRECTORY environment variable not set")
# Ensure the directory exists
os.makedirs(csv_directory, exist_ok=True)

#fred info
# Ensure the urllib uses the certifi certificate bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Set up FRED API key (you need to sign up at https://fred.stlouisfed.org/ to get an API key)
fred_key = os.getenv('fred_key')
fred = Fred(api_key=fred_key)



def Create_sp500_csv():
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
    
    # # Save the DataFrame to a CSV file for reference
    save_csv(df, 'sp500_df.csv')


def create_crypto_csv():

    crypto_df = pd.DataFrame({
            "Symbol": ["AAVE/USD","AVAX/USD","BAT/USD","BCH/USD","BTC/USD","CRV/USD","DOGE/USD","DOT/USD","ETH/USD","LINK/USD","LTC/USD","MKR/USD","SHIB/USD","SUSHI/USD","UNI/USD","USDC/USD","USDT/USD","XTZ/USD"]
        })# "GRT/USD"
    
    # Save the DataFrame to a CSV file
    save_csv(crypto_df, 'crypto_df.csv')

def save_csv(df, filename):
    file_path = os.path.join(csv_directory, filename)
    df.to_csv(file_path, index=False)


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