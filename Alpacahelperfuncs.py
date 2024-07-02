import alpaca_trade_api as tradeapi
import numpy as np


def get_available_balance(api, symbol):
    try:
        position = api.get_position(symbol)
        return float(position.qty)
    except tradeapi.rest.APIError as e:
        if e.status_code == 404:
            return 0.0
        else:
            raise e

def get_market_value(api, symbol):
    try:
        market_value = api.get_position(symbol).market_value
        return market_value
    except tradeapi.rest.APIError as e:
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
    

def execute_trade(order_type, amount, symbol, api):
    try:
        if order_type == 'buy':
            order = api.submit_order(
                symbol=symbol,
                qty=amount,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
        elif order_type == 'sell':
            available_balance = get_available_balance(symbol)
            if available_balance < amount:
                amount = available_balance
            order = api.submit_order(
                symbol=symbol,
                qty=amount,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
        print(f"Successfully Executed {order_type} for ",amount, " Bitcoin") # order: {order} <- include if you want all the order data as well
    except Exception as e:
        print(f"Error executing {order_type} order: {e}")
