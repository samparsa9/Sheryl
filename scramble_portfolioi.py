import os
import src.utils.alpaca_utils as hf
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import random

load_dotenv()
# Alpaca Info
api_key = os.getenv('api_key')
api_secret = os.getenv("api_secret")
base_url = os.getenv('base_url')

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


def main():
    
    api = tradeapi.REST(api_key, api_secret, 'https://paper-api.alpaca.markets/', api_version='v2')
    throw_off_portfolio(api, 3, 3, ["AAVE/USD","AVAX/USD","BAT/USD","BCH/USD","BTC/USD","CRV/USD","DOGE/USD","DOT/USD","ETH/USD","LINK/USD","LTC/USD","MKR/USD","SHIB/USD","SUSHI/USD","UNI/USD","USDC/USD","USDT/USD","XTZ/USD"], True)
    pass
if __name__ == "__main__":
    main()
