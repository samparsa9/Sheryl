import alpaca_trade_api as tradeapi
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.settings import API_KEY, API_SECRET, BASE_URL

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def get_available_balance(symbol, crypto=False):
    try:
        if crypto:
            symbol = symbol.replace("-", "") 
            symbol = symbol.replace("/", "")
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

def get_positions():
    return api.list_positions()

def cluster_and_allocation_setup(starting_cash, df, num_clusters, crypto=False):
    portfolio_amount = starting_cash

    # Setup the initial cluster information dataframe
    cluster_info_df = pd.DataFrame({
        "Cluster": [i for i in range(num_clusters)],
        "Op Percentage": [0.25, 0.25, 0.25, 0.25],
        "Op $ In Cluster": [0.0] * num_clusters,
        "Num Stocks": [0.0] * num_clusters,
        "$ Per Stock": [0.0] * num_clusters,
        "Tickers": [[] for _ in range(num_clusters)]
    })
    # Ensure the "Op Percentage" column is of float type
    cluster_info_df["Op Percentage"] = cluster_info_df["Op Percentage"].astype(float)

    # Ensure portfolio_amount is a float
    portfolio_amount = float(portfolio_amount)

    # Perform the calculation and rounding
    cluster_info_df["Op $ In Cluster"] = round(cluster_info_df["Op Percentage"] * portfolio_amount, 2)


    # Populate the tickers and number of stocks for each cluster
    for ticker, row in df.iterrows():
        cluster = row["Cluster"]
        cluster_info_df.loc[cluster, "Tickers"].append(ticker)
        cluster_info_df.loc[cluster, "Num Stocks"] += 1

    for cluster, row in cluster_info_df.iterrows():
        # Calculate the amount per stock for each cluster
        if cluster_info_df.loc[cluster, "Num Stocks"] > 0:
            cluster_info_df.loc[cluster, "$ Per Stock"] = round((cluster_info_df.loc[cluster, "Op $ In Cluster"] / cluster_info_df.loc[cluster, "Num Stocks"]), 2)
        else:
            cluster_info_df.loc[cluster, "$ Per Stock"] = 0  # Avoid division by zero

    # Setup the current portfolio allocation dataframe
    current_portfolio_allocation = pd.DataFrame({
        "Cluster": [i for i in range(num_clusters)],
        "Cur Pct Alloc": [0.0] * num_clusters,
        "Pct Off Op": [0.0] * num_clusters,
        "Cur $ In Cluster": [0.0] * num_clusters
    })
    current_portfolio_allocation.set_index("Cluster", inplace=True)

    # Calculate the current dollar allocation in each cluster
    positions = get_positions()
    
    position_data = {position.symbol.replace("-",""): float(position.market_value) for position in positions}
    for cluster in cluster_info_df.index:
        dollars_in_this_cluster = 0.0
        tickers = cluster_info_df.loc[cluster, "Tickers"]
        for ticker in tickers:
            ticker = ticker.replace("-","")
            try:
                market_value = position_data[ticker]
            except:
                market_value = 0
            dollars_in_this_cluster += market_value
        current_portfolio_allocation.loc[cluster, "Cur $ In Cluster"] = round(dollars_in_this_cluster, 2)

    # Calculate the percentage allocation and the deviation from the optimal allocation
    account_value = get_total_account_value()
    for cluster in cluster_info_df.index:
        dollars_in_this_cluster = current_portfolio_allocation.loc[cluster, 'Cur $ In Cluster']
        current_cluster_pct_allocation = (dollars_in_this_cluster / account_value)
        optimal_cluster_pct_allocation = cluster_info_df.loc[cluster, "Op Percentage"]

        current_portfolio_allocation.loc[cluster, "Cur Pct Alloc"] = current_cluster_pct_allocation
        current_portfolio_allocation.loc[cluster, "Pct Off Op"] = current_cluster_pct_allocation - optimal_cluster_pct_allocation

    # Combine the optimal and current allocation dataframes
    combined_df = cluster_info_df.join(current_portfolio_allocation)

    return combined_df