import sys
import os
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta, timezone
import time as tm
import sys
from dotenv import load_dotenv
import Alpacahelperfuncs as hf
import pandas as pd
import JOKR_setup as setup

load_dotenv()
api_key = os.getenv('api_key')
api_secret = os.getenv("api_secret")
base_url = os.getenv('base_url')

# Initialize Alpaca API
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')


def Get_current_portfolio_allocation(optimal_portfolio_allocation_df, api):
    # Create a new dataframe that represents our current portfolios allocation among clusters
    current_portfolio_allocation = pd.DataFrame({
            "Cluster": [0,1,2,3,4],
            "Dollars In Cluster": [0] * 5,
        })
    current_portfolio_allocation.set_index("Cluster", inplace=True)
    # This for loop will be used to calulate the total dollars in each cluster by looping through each ticker in each cluster
    # and adding the market value of our position in that ticker to a running sum
    for cluster in optimal_portfolio_allocation_df.index:
        dollars_in_this_cluster = 0
        for ticker in optimal_portfolio_allocation_df.at[cluster, "Tickers"]:
            dollars_in_this_cluster += hf.get_market_value(api, ticker)
        # Populating the Dollars In Cluster column with these new values in our current portfolio allocation df
        current_portfolio_allocation.at[cluster, "Dollars In Cluster"] = dollars_in_this_cluster
    return current_portfolio_allocation

def Get_most_unoptimized_clusters(optimal_portfolio_allocation_df, current_portfolio_allocation, api):
    """
    This function will take in the cluster information dataframe and the api,
    it will then see if the portfolio is balanced correctly depending on the cluster allocations,
    if it is not, it will return the two clusters off by the most percentage
    """
    
    Highest_unoptimal_allocation_pct = -float('inf')
    Lowest_unoptimal_allocation_pct = float('inf')
    Highest_unoptimal_allocation_cluster = 0
    Lowest_unoptimal_allocation_cluster = 0

    for index, row in optimal_portfolio_allocation_df.iterrows():
        current_dollar_allocation = current_portfolio_allocation.at[index, "Dollars In Cluster"]
        optimal_dollar_allocation = optimal_portfolio_allocation_df.at[index, "Dollars In Cluster"]

        if optimal_dollar_allocation != 0:
            pct_diff_between_current_and_optimal_allocation = (current_dollar_allocation / optimal_dollar_allocation) - 1

            if pct_diff_between_current_and_optimal_allocation > Highest_unoptimal_allocation_pct:
                Highest_unoptimal_allocation_pct = pct_diff_between_current_and_optimal_allocation
                Highest_unoptimal_allocation_cluster = index
            elif pct_diff_between_current_and_optimal_allocation < Lowest_unoptimal_allocation_pct:
                Lowest_unoptimal_allocation_pct = pct_diff_between_current_and_optimal_allocation
                Lowest_unoptimal_allocation_cluster = index

    tuple_to_return = ((Highest_unoptimal_allocation_cluster, Highest_unoptimal_allocation_pct), (Lowest_unoptimal_allocation_cluster, Lowest_unoptimal_allocation_pct))
    return tuple_to_return
        

def Is_balanced(H_pct, L_pct):
    return abs(H_pct) < 0.03 and abs(L_pct) < 0.03



def main():
    # When we run this algo for the very first time we won't be in a position
    in_position = hf.in_position(api)

    # Specifying the location of where we want our sp500 csv to be loaded, or if it already exists, where it is to update it
    location_of_sp500_csv_file = 'Sheryl/sp500_companies.csv'

    # Running wikapedia scraper to populate initial sp500 csv with ticker symbols of sp500 companies
    setup.Create_sp500_csv(location_of_sp500_csv_file)

    # Setting our stockdf to be the csv file we just created
    stockdf = pd.read_csv('Sheryl/sp500_companies.csv', index_col='Symbol')

    # Creating our tickers list which we will pass into the Calculate_features function
    tickers = setup.Create_list_of_tickers(stockdf.index)

    # Calculating our features for each ticker and populating the dataframe with these values
    setup.Calculate_features(tickers, stockdf)

    # Dropping any columns that may have resulted in NaN values
    stockdf = stockdf.dropna()

    # Creating a scaled_data numpy array that we will pass into our K means algorithm
    scaled_data = setup.Scale_data(stockdf)

    # Running k means
    setup.Apply_K_means(stockdf, scaled_data)

    # Soring the data frame based on cluster value and saving these values to the dataframe, and then updating the csv
    setup.Sort_and_save(stockdf, location_of_sp500_csv_file)

    # Creating a new dataframe that will contain information about the clusters that will be used for trading logic
    optimal_portfolio_allocation_info_df = setup.cluster_df_setup(1000000, stockdf)
    print(optimal_portfolio_allocation_info_df)

    # Retreive our current portflio cluster allocation
    current_portfolio_allocation_df = Get_current_portfolio_allocation(optimal_portfolio_allocation_info_df, api)

    # Find what clusters are unoptimized and by how much
    returned_tuple = Get_most_unoptimized_clusters(optimal_portfolio_allocation_info_df, current_portfolio_allocation_df, api)
    
    # Save unoptimized clusters and percentages
    H_unop_alloc_cluster = returned_tuple[0][0]
    H_unop_alloc_pct = returned_tuple[0][1]
    L_unop_alloc_cluster = returned_tuple[1][0]
    L_unop_alloc_pct = returned_tuple[1][1]

    
    
    # # Plotting cluster data
    # setup.plot_clusters(stockdf)

    
    while True: # FOR DEBUGGING, SET BACK TO TRUE
        try:
            print("----------New Clustering and Allocation Iteration---------")
            if not in_position and (datetime.now(timezone.utc).hour == 14 and datetime.now(timezone.utc).minute == 00):
                #need to buy in proportionally to each cluster
                for index, row in optimal_portfolio_allocation_info_df.iterrows():
                    stocks_for_this_cluster = optimal_portfolio_allocation_info_df.at[index, "Tickers"]
                    dollars_per_stock_for_cluster = optimal_portfolio_allocation_info_df.at[index, "Amount Per Stock"]
                    for stock in stocks_for_this_cluster:
                        hf.execute_trade("buy", dollars_per_stock_for_cluster, stock, api)
                in_position = True
            if in_position and (datetime.now(timezone.utc).hour == 14 and datetime.now(timezone.utc).minute == 00):
                while not Is_balanced(H_unop_alloc_pct, L_unop_alloc_pct):
                    # buy random tickers in L_unop_alloc_cluster and sell random tickers in H_unop_alloc_cluster
                    lowest_market_value = float('inf')
                    ticker_to_buy = None
                    for ticker in optimal_portfolio_allocation_info_df.at[L_unop_alloc_cluster, "Tickers"]:
                        market_value = hf.get_market_value(api, ticker)
                        if market_value < lowest_market_value:
                            lowest_market_value = market_value
                            ticker_to_buy = ticker
                    if ticker_to_buy:
                        hf.execute_trade("buy", lowest_market_value, ticker_to_buy, api)
                    
                
                    highest_market_value = -float('inf')
                    ticker_to_sell = None
                    for ticker in optimal_portfolio_allocation_info_df.at[H_unop_alloc_cluster, "Tickers"]:
                        market_value = hf.get_market_value(api, ticker)
                        if market_value < lowest_market_value:
                            highest_market_value = market_value
                            ticker_to_sell = tickers
                    if ticker_to_sell:
                        hf.execute_trade("sell", highest_market_value, ticker_to_sell, api)

            def calculate_seconds_till_next_reallocation():
                now = datetime.now(timezone.utc)
                target_time = now.replace(hour=14, minute=0, second=0, microsecond=0)
                if now > target_time:
                    target_time += timedelta(days=1)
                return int((target_time - now).total_seconds())
            
            seconds_left_till_next_reallocation = calculate_seconds_till_next_reallocation()

            for seconds_left in range(seconds_left_till_next_reallocation, 0, -1):
                hours_left = seconds_left // 3600
                minutes_left = (seconds_left % 3600) // 60
                remaining_seconds = seconds_left % 60
                sys.stdout.write(f"\r{hours_left} hours, {minutes_left} minutes, {remaining_seconds} seconds till next iteration!!!")
                sys.stdout.flush()
                tm.sleep(1)
            print("It's 14:00 UTC, time for reallocation!")
        except Exception as e:
            print(f"Error in main loop: {e}")
            tm.sleep(60)  # Wait for 1 minute before retrying
    

if __name__ == "__main__":
    main()