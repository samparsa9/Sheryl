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


def main():

    while True: # FOR DEBUGGING, SET BACK TO TRUE
        try:
            print("----------New Clustering and Allocation Iteration---------")
            # When we run this algo for the very first time we won't be in a position, otherwise we will be
            in_position = hf.in_position(api)
         
            # If the time is 10:00 AM, set up all data that will be needed for allocation/reallocation
            if (datetime.now(timezone.utc).hour == 14 and datetime.now(timezone.utc).minute == 00):
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
                current_portfolio_allocation_df = setup.Get_current_portfolio_allocation(optimal_portfolio_allocation_info_df, api)

                # Find what clusters are unoptimized and by how much
                returned_tuple = setup.Get_most_unoptimized_clusters(optimal_portfolio_allocation_info_df, current_portfolio_allocation_df, api)
                
                # Save unoptimized clusters and percentages
                H_unop_alloc_cluster = returned_tuple[0][0]
                H_unop_alloc_pct = returned_tuple[0][1]
                L_unop_alloc_cluster = returned_tuple[1][0]
                L_unop_alloc_pct = returned_tuple[1][1]

                # # Plotting cluster data
                # setup.plot_clusters(stockdf)


                if not in_position:
                    # For every cluster
                    for index, row in optimal_portfolio_allocation_info_df.iterrows():
                        # Create a list of the tickers
                        stocks_for_this_cluster = optimal_portfolio_allocation_info_df.at[index, "Tickers"]
                        # Store how much we should buy of each stock
                        dollars_per_stock_for_cluster = optimal_portfolio_allocation_info_df.at[index, "Amount Per Stock"]
                        # For every stock in this cluster
                        for stock in stocks_for_this_cluster:
                            # Buy the specified amount of that stock
                            hf.execute_trade("buy", dollars_per_stock_for_cluster, stock, api)
                    in_position = True
                if in_position:
                    while not setup.Is_balanced(H_unop_alloc_pct, L_unop_alloc_pct):
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