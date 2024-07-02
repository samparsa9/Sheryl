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
import pytz
import traceback

load_dotenv()
# Alpaca Info
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

            # Get the current time in EST
            est = pytz.timezone('US/Eastern')
            now = datetime.now(est)

            # The Hour and Minute at which we want our bot to check allocations
            hour_to_trade = int(now.hour) # SET BACK
            minute_to_trade = int(now.minute) # SET BACK

            # If the time is 10:00 AM, set up all data that will be needed for allocation/reallocation
            if (now.hour == hour_to_trade and now.minute == minute_to_trade): # CHANGE THE TIME TO DEBUG
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

                # Save the newsly created cluster df to a csv file
                setup.Save_cluster_df(optimal_portfolio_allocation_info_df, 'Sheryl/sp500_cluster_info.csv')

                
                

                # # Plotting cluster data
                # setup.plot_clusters(stockdf)


                if not in_position:
                    # For every cluster
                    for index, row in optimal_portfolio_allocation_info_df.iterrows():
                        # Create a list of the tickers
                        stocks_for_this_cluster = optimal_portfolio_allocation_info_df.at[index, "Tickers"]
                        # Store how much we should buy of each stock
                        dollars_per_stock_for_cluster = int(optimal_portfolio_allocation_info_df.at[index, "Amount Per Stock"])
                        # For every stock in this cluster
                        for stock in stocks_for_this_cluster:
                            # Buy the specified amount of that stock
                            hf.execute_trade("buy", dollars_per_stock_for_cluster, stock, api, notional=True)
                            
                    in_position = True
                    setup.send_email("Entered Initial Positions", " ")

                if in_position:
                    current_portfolio_df = setup.Get_current_portfolio_allocation(optimal_portfolio_allocation_info_df, api)
                    while not setup.Is_balanced(optimal_portfolio_allocation_info_df,current_portfolio_df, api):
                        unop_clusters = setup.Get_most_unoptimized_clusters(optimal_portfolio_allocation_info_df, current_portfolio_df, api)
                        H_unop_alloc_cluster = unop_clusters[0][0]
                        H_unop_alloc_pct = unop_clusters[0][1]
                        L_unop_alloc_cluster = unop_clusters[1][0]
                        L_unop_alloc_pct = unop_clusters[1][1]
                        print(f"Portfolio is not balanced: Highest unoptimal cluster: {H_unop_alloc_cluster} by {H_unop_alloc_pct}")
                        print(f"Portfolio is not balanced: Lowest unoptimal cluster: {L_unop_alloc_cluster} by {L_unop_alloc_pct}")
                        # buy random tickers in L_unop_alloc_cluster and sell random tickers in H_unop_alloc_cluster
                        lowest_market_value = float('inf')
                        ticker_to_buy = None
                        for ticker in optimal_portfolio_allocation_info_df.at[L_unop_alloc_cluster, "Tickers"]:
                            market_value = float(hf.get_market_value(api, ticker))
                            if market_value < lowest_market_value:
                                lowest_market_value = market_value
                                ticker_to_buy = ticker
                        if ticker_to_buy:
                            hf.execute_trade("buy", int(lowest_market_value), ticker_to_buy, api, notional=True)
                        
                    
                        highest_market_value = -float('inf')
                        ticker_to_sell = None
                        for ticker in optimal_portfolio_allocation_info_df.at[H_unop_alloc_cluster, "Tickers"]:
                            market_value = float(hf.get_market_value(api, ticker))
                            if market_value < lowest_market_value:
                                highest_market_value = market_value
                                ticker_to_sell = ticker
                        if ticker_to_sell:
                            hf.execute_trade("sell", int(highest_market_value), ticker_to_sell, api, notional=True)
                    setup.send_email('Portfolio Rebalancing Complete', ' ')
                    current_portfolio_df = setup.Get_current_portfolio_allocation(optimal_portfolio_allocation_info_df, api)
                else:
                    setup.send_email('Portfolio is Still Balanced', ' ')

            def calculate_seconds_till_next_reallocation():
                now = datetime.now(est)
                target_time = now.replace(hour=hour_to_trade, minute=minute_to_trade, second=0, microsecond=0)
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
            print("It's time for reallocation!")
        except Exception as e:
            print(f"Error in main loop: {e}")
            traceback.print_exc()
            tm.sleep(60)  # Wait for 1 minute before retrying
    

if __name__ == "__main__":
    main()