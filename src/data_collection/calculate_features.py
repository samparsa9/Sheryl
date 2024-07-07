import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred

def calculate_features(symbols, df, fred_api_key, batch_size=10, crypto=False):
    fred = Fred(api_key=fred_api_key)
    BTC_data = yf.download('BTC-USD', period='1y')
    BTC_data['Market Return'] = BTC_data['Adj Close'].pct_change()

    series_id = 'GS1'
    treasury_data = fred.get_series(series_id)
    risk_free_rate = treasury_data.iloc[-1] / 100 if not treasury_data.empty else 0

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        if crypto:
            batch = [symbol.replace("/", "-") for symbol in batch]
        batch_data = yf.download(batch, period='1y', group_by='ticker')

        for ticker in batch:
            ticker_data = batch_data[ticker]
            if isinstance(ticker_data, pd.DataFrame):
                ticker_data = ticker_data.copy()
                ticker_data['Return'] = ticker_data['Adj Close'].pct_change()
                df.at[ticker, "Daily $ Volume"] = ticker_data['Volume'].iloc[-1]

                calculate_moving_averages(ticker_data, df, ticker)
                calculate_beta_sharpe(ticker_data, df, ticker, BTC_data, risk_free_rate)
                calculate_rsi(ticker_data, df, ticker)
    
def calculate_moving_averages(ticker_data, df, ticker):
    latest_close = ticker_data['Adj Close'].iloc[-1]
    if len(ticker_data) >= 50:
        ticker_data['50 SMA'] = ticker_data['Adj Close'].rolling(window=50).mean()
        df.at[ticker, '50 SMA % Difference'] = ((latest_close - ticker_data['50 SMA'].iloc[-1]) / ticker_data['50 SMA'].iloc[-1]) * 100

    if len(ticker_data) >= 200:
        ticker_data['200 SMA'] = ticker_data['Adj Close'].rolling(window=200).mean()
        df.at[ticker, '200 SMA % Difference'] = ((latest_close - ticker_data['200 SMA'].iloc[-1]) / ticker_data['200 SMA'].iloc[-1]) * 100

def calculate_beta_sharpe(ticker_data, df, ticker, BTC_data, risk_free_rate):
    returns = pd.concat([ticker_data['Return'], BTC_data['Market Return']], axis=1).dropna()
    covariance = np.cov(returns['Return'], returns['Market Return'])[0, 1]
    BTC_variance = np.var(returns['Market Return'])
    df.at[ticker, 'Beta value'] = covariance / BTC_variance if BTC_variance != 0 else np.nan

    daily_risk_free_rate = risk_free_rate / 252
    ticker_data['Excess Return'] = ticker_data['Return'] - daily_risk_free_rate
    mean_excess_return = ticker_data['Excess Return'].mean()
    std_excess_return = ticker_data['Excess Return'].std()
    df.at[ticker, 'Sharpe Ratio'] = (mean_excess_return / std_excess_return) * np.sqrt(252) if std_excess_return != 0 else np.nan
    df.at[ticker, 'Volatility'] = ticker_data['Return'].std() * np.sqrt(252)


def calculate_rsi(ticker_data, df, ticker):
# Relative Strength Index (RSI)
    delta = ticker_data['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    ticker_data['RSI'] = 100 - (100 / (1 + RS))
    df.at[ticker, 'RSI'] = ticker_data['RSI'].iloc[-1]