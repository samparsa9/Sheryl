import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred

# Function to calculate desired features for each ticker that will be used to put them in clusters
def calculate_features(symbols, df, fred_api_key, batch_size=10, crypto=False):
    # Initializing Fred instance
    fred = Fred(api_key=fred_api_key)
    # Retreiving 1 yr of bitcoin data to be used later on
    BTC_data = yf.download('BTC-USD', period='1y') 
    # Creating a new column named "Market Return" which will represent the day-to-day change of the Adj Close
    BTC_data['Market Return'] = BTC_data['Adj Close'].pct_change()

    # Getting the information from Fred to calculate risk free rate
    series_id = 'GS1'
    treasury_data = fred.get_series(series_id)
    # Turning the risk free rate from a 
    risk_free_rate = treasury_data.iloc[-1] / 100 if not treasury_data.empty else 0

    # We will process the list of symbols in batches and calculate features for each symbol one at a time
    for i in range(0, len(symbols), batch_size):
        # Getting the current batch
        batch = symbols[i:i + batch_size]
        # If our symbols are crypto we will replace "/" with "-" so yfinance will recognize the symbol
        if crypto:
            batch = [symbol.replace("/", "-") for symbol in batch]
        # Download 1yr data for the symbols in the batch
        batch_data = yf.download(batch, period='1y', group_by='ticker')

        # For every ticker in this batch, calculate its features and add their values to their corresponding column
        # in the dataframe that is passed into this function
        for ticker in batch:
            # Index the batch_data to get the historical data for our specific symbol
            ticker_data = batch_data[ticker]

            # If ticker_data is a dataframe (it is), make a copy to avoid copy warnings
            if isinstance(ticker_data, pd.DataFrame):
                ticker_data = ticker_data.copy()
                # Create a new column that will represent the day-to-day price change for the symbol for each day
                ticker_data['Return'] = ticker_data['Adj Close'].pct_change()
                # Put the most recent volume data in the  "Daily $ Volume" cell for this specific symbol
                df.at[ticker, "Daily $ Volume"] = ticker_data['Volume'].iloc[-1]

                # Call our other feature creation functions
                calculate_moving_averages(ticker_data, df, ticker)
                calculate_beta_sharpe(ticker_data, df, ticker, BTC_data, risk_free_rate)
                calculate_rsi(ticker_data, df, ticker)
    
# Function to calculate % above or below 50 and 200 SMAs
def calculate_moving_averages(ticker_data, df, ticker):
    # Setting the latest close to be the Adj Close in the last row of the ticker_data df
    latest_close = ticker_data['Adj Close'].iloc[-1]
    # If the length of the df is greater than 50, we can create a 50 SMA
    if len(ticker_data) >= 50:
        # The value for the 50 SMA will be a rolling mean of the last 50 close prices
        ticker_data['50 SMA'] = ticker_data['Adj Close'].rolling(window=50).mean()
        # Populate the cell for the specific symbol to be the % higher or lower the latest close is
        # compared to the 50 SMA
        df.at[ticker, '50 SMA % Difference'] = ((latest_close - ticker_data['50 SMA'].iloc[-1]) / ticker_data['50 SMA'].iloc[-1]) * 100

    # If the length of the df is greater than 200, we can create a 200 SMA
    if len(ticker_data) >= 200:
        # The value for the 200 SMA will be a rolling mean of the last 200 close prices
        ticker_data['200 SMA'] = ticker_data['Adj Close'].rolling(window=200).mean()
        # Populate the cell for the specific symbol to be the % higher or lower the latest close is
        # compared to the 200 SMA
        df.at[ticker, '200 SMA % Difference'] = ((latest_close - ticker_data['200 SMA'].iloc[-1]) / ticker_data['200 SMA'].iloc[-1]) * 100

# Function to calculate beta and sharpe ratios
def calculate_beta_sharpe(ticker_data, df, ticker, BTC_data, risk_free_rate):
    # Merge the symbol returns column to the BTC returns column
    returns = pd.concat([ticker_data['Return'], BTC_data['Market Return']], axis=1).dropna()
    # Calculate the covariange
    covariance = np.cov(returns['Return'], returns['Market Return'])[0, 1]
    # Calculate Bitcoin variance
    BTC_variance = np.var(returns['Market Return'])
    # Populate the cell for the symbol's beta value as the cov / btc var
    df.at[ticker, 'Beta value'] = covariance / BTC_variance if BTC_variance != 0 else np.nan

    # Calculate the daily risk free rate
    daily_risk_free_rate = risk_free_rate / 252
    # Calculate excess returns as each days return - the daily risk free rate
    ticker_data['Excess Return'] = ticker_data['Return'] - daily_risk_free_rate
    # Take the mean of this mean of the Excess Return column
    mean_excess_return = ticker_data['Excess Return'].mean()
    # Take the standard deviation of the Excess Return column
    std_excess_return = ticker_data['Excess Return'].std()
    # Populate the symbol's Sharpe ration with the mean excess return / std excess return * sqrt(252)
    df.at[ticker, 'Sharpe Ratio'] = (mean_excess_return / std_excess_return) * np.sqrt(252) if std_excess_return != 0 else np.nan
    # Also calculate the symbols volatility and put it in the correct cell
    df.at[ticker, 'Volatility'] = ticker_data['Return'].std() * np.sqrt(252)

# Function to calculate the Relative Strength Index (RSI) for the symbol
def calculate_rsi(ticker_data, df, ticker):
    # delta will be equal to the difference between two days closing prices
    delta = ticker_data['Adj Close'].diff()
    # gains will be anywhere where this difference is positive, and use it to create a rolling 14 period mean
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    # gains will be anywhere where this difference is negative, and use it to create a rolling 14 period mean
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # relative strength is the average gain / average loss
    RS = gain / loss
    # Calculate the RSI and populate the corresponding cell
    ticker_data['RSI'] = 100 - (100 / (1 + RS))
    df.at[ticker, 'RSI'] = ticker_data['RSI'].iloc[-1]