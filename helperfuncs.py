import alpaca_trade_api as tradeapi
import numpy as np
import BTC2hrstrat as strat



def get_available_balance(asset='BTC/USD'):
    try:
        position = strat.api.get_position(asset)
        return float(position.qty)
    except tradeapi.rest.APIError as e:
        if e.status_code == 404:
            return 0.0
        else:
            raise e



def execute_trade(order_type, amount):
    try:
        if order_type == 'buy':
            order = strat.api.submit_order(
                symbol=strat.symbol,
                qty=amount,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
        elif order_type == 'sell':
            available_balance = get_available_balance(strat.symbol)
            if available_balance < amount:
                amount = available_balance
            order = strat.api.submit_order(
                symbol=strat.symbol,
                qty=amount,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
        print(f"Successfully Executed {order_type} for ",amount, " Bitcoin") # order: {order} <- include if you want all the order data as well
    except Exception as e:
        print(f"Error executing {order_type} order: {e}")
