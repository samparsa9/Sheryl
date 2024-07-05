import yfinance as yf
import pandas as pd
import numpy as np
import requests

def calculate_sharpe_ratio(ticker='BTC-USD', risk_free_rate=0.053):
    # Download BTC-USD historical data for the past year
    btc_data = yf.download(ticker, period='1y')
    
    # Calculate daily returns
    btc_data['Daily Return'] = btc_data['Adj Close'].pct_change()
    
    # Calculate excess returns by subtracting the daily risk-free rate (annual rate divided by 252 trading days)
    daily_risk_free_rate = risk_free_rate / 252
    btc_data['Excess Return'] = btc_data['Daily Return'] - daily_risk_free_rate
    
    # Calculate the mean and standard deviation of excess returns
    mean_excess_return = btc_data['Excess Return'].mean()
    std_excess_return = btc_data['Excess Return'].std()
    
    # Calculate the Sharpe ratio (annualized)
    sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(252)
    
    return sharpe_ratio

# sharpe_ratio = calculate_sharpe_ratio()
# print(f"The Sharpe Ratio for Bitcoin (BTC-USD) over the past year is: {sharpe_ratio:.2f}")
df = yf.download("BTC-USD", period='1y', group_by='ticker')
print(df.loc['2024-06-30', 'Volume'] / df.loc['2024-06-30', 'Close'])



def get_crypto_market_cap(symbol):
    # Map some common cryptocurrency symbols to CoinGecko IDs
    symbol_map = {
        'BTC-USD': 'bitcoin',
        'ETH-USD': 'ethereum',
        'AAVE-USD': 'aave',
        'AVAX-USD': 'avalanche-2',
        'BAT-USD': 'basic-attention-token',
        'BCH-USD': 'bitcoin-cash',
        'CRV-USD': 'curve-dao-token',
        'DOGE-USD': 'dogecoin',
        'DOT-USD': 'polkadot',
        'LINK-USD': 'chainlink',
        'LTC-USD': 'litecoin',
        'MKR-USD': 'maker',
        'SHIB-USD': 'shiba-inu',
        'SUSHI-USD': 'sushi',
        'UNI-USD': 'uniswap',
        'USDC-USD': 'usd-coin',
        'USDT-USD': 'tether',
        'XTZ-USD': 'tezos'
    }
    
    # Convert to lowercase and look up in the symbol_map
    symbol_id = symbol_map.get(symbol.upper())
    if not symbol_id:
        print(f"Symbol {symbol} not found in the symbol map.")
        return None

    url = f'https://api.coingecko.com/api/v3/coins/{symbol_id}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['market_data']['market_cap']['usd']
    else:
        print(f"Failed to fetch market cap for {symbol}. HTTP Status code: {response.status_code}")
        return None

# Example usage
crypto_symbol = 'BTC-USD'
market_cap = get_crypto_market_cap(crypto_symbol)
print(f"Market Cap of {crypto_symbol}: {market_cap}")

