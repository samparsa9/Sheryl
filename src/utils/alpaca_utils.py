import alpaca_trade_api as tradeapi

import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.settings import API_KEY, API_SECRET, BASE_URL
from dotenv import load_dotenv

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def get_available_balance(symbol):
    try:
        position = api.get_position(symbol)
        return float(position.qty)
    except tradeapi.rest.APIError as e:
        if e.status_code == 404:
            return 0.0
        else:
            raise e

def get_market_value(ticker, crypto=False):
    try:
        ticker = ticker.replace("-", "") # not sure if this should be here
        ticker = ticker.replace("/", "")
        position = api.get_position(ticker)
        market_value = float(position.market_value)
        return market_value
    except Exception as e:
        print(f"Failed to get market value for ticker '{ticker}': {e}")
        return 0

def in_position():
    try:
        positions = api.list_positions()
        if positions == []:
            return False
        else:
            return True
    except tradeapi.rest.APIError as e:
        print("Error fetching positions: {e}")
        return []
    
def execute_trade(action, amount, symbol, notional=False, crypto=False):
    symbol = symbol.replace("-", "") # not sure if this should be here
    symbol = symbol.replace("/", "")
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
            order = api.submit_order(
                symbol=str(symbol),
                notional=float(amount),
                side=action,
                type='market',
                time_in_force='gtc'
            )
        elif not notional and crypto == False:
            order = api.submit_order(
                symbol=str(symbol),
                qty=float(amount),
                side=action,
                type='market',
                time_in_force='day'
            )
        elif not notional and crypto == True:
            order = api.submit_order(
                symbol=str(symbol),
                qty=float(amount),
                side=action,
                type='market',
                time_in_force='gtc'
            )
        if notional:
            print(f"Successfully Executed {action} for ${amount} worth of {symbol}")
        else:
            print(f"Successfully Executed {action} for {amount} shares of {symbol}")
        return order
    except Exception as e:
        print(f"Error executing {action} order for {symbol}: {e}")

def get_total_account_value():
    account = api.get_account()
    return float(account.equity) 

def get_cost_basis(symbol):
    symbol = symbol.replace("-", "") 
    symbol = symbol.replace("/", "")
    try:
        # Retrieve the specific position
        position = api.get_position(symbol)
        
        # Calculate the cost basis
        cost_basis = float(position.avg_entry_price) * float(position.qty)
        print(f"Symbol: {symbol}, Cost Basis: ${cost_basis:.2f}")
    except tradeapi.rest.APIError as e:
        print(f"Error retrieving position for {symbol}: {e}")
    return cost_basis

def get_buying_power():
    return api.get_account().buying_power

