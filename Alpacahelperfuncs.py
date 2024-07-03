import alpaca_trade_api as tradeapi
import numpy as np
import os
from dotenv import load_dotenv
import JOKR_strat as js

# load_dotenv()
# # Alpaca Info
# api_key = os.getenv('api_key')
# api_secret = os.getenv("api_secret")
# base_url = os.getenv('base_url')


# # Initialize Alpaca API
# api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

def get_available_balance(api, symbol):
    try:
        position = api.get_position(symbol)
        return float(position.qty)
    except tradeapi.rest.APIError as e:
        if e.status_code == 404:
            return 0.0
        else:
            raise e

def get_market_value(api, ticker, crypto=False):
    try:
        ticker = ticker.replace("-", "") # not sure if this should be here
        position = api.get_position(ticker)
        market_value = float(position.market_value)
        return market_value
    except Exception as e:
        print(f"Failed to get market value for ticker '{ticker}': {e}")
        return 0

def in_position(api):
    try:
        positions = api.list_positions()
        if positions == []:
            return False
        else:
            return True
    except tradeapi.rest.APIError as e:
        print("Error fetching positions: {e}")
        return []
    
def execute_trade(action, amount, symbol, api, notional=False, crypto=False):
    try:
        if notional and crypto == False:
            order = api.submit_order(
                symbol=str(symbol),
                notional=float(amount),
                side=action,
                type='market',
                time_in_force='day'
            )
        elif notional and crypto == True:
            print(f"trade amount is {amount}")
            # symbol = symbol.replace("-", "") # added this line
            order = api.submit_order(
                symbol=str(symbol),
                notional=float(amount),
                side=action,
                type='market',
                time_in_force='gtc'
            )
        else:
            order = api.submit_order(
                symbol=str(symbol),
                qty=int(amount),
                side=action,
                type='market',
                time_in_force='gtc'
            )
        print(f"Successfully Executed {action} for ${amount} of {symbol}")
        return order
    except Exception as e:
        print(f"Error executing {action} order: {e}")

def get_total_account_value(api):
    account = api.get_account()
    return float(account.equity) # Assuming equity represents the total market value of your account

