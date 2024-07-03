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
    ############ STATIC STUFF AMONG ITERATIONS ################
    ############ ONLY BEING RUN FIRST TIME WE EVER LAUNCH ALGO #############

    # Get the current time in EST
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)

    # The Hour and Minute at which we want our bot to check allocations
    hour_to_trade = int(now.hour) # SET BACK
    minute_to_trade = int(now.minute) # SET BACK

    # Location of our main info dataframe
    location_of_main_csv_file = 'Sheryl/crypto_data.csv' #Change this to change what symbol we are looking at

    # Location of our cluster info dataframe
    location_of_cluster_csv_file = 'Sheryl/sp500_cluster_info'

    # SET BACK TO 5
    num_clusters = 4 # Use this to specify how many cluster for K-means

    # Set true since were dealing with crypto symbols
    crypto = True # SET BACK TO FALSE

    while True: # FOR DEBUGGING, SET BACK TO TRUE
        try:
            print("----------New Clustering and Allocation Iteration---------")

            # If the time is 10:00 AM, set up all data that will be needed for allocation/reallocation
            if (now.hour == hour_to_trade and now.minute == minute_to_trade): # CHANGE THE TIME TO DEBUG
                # When we run this algo for the very first time we won't be in a position, otherwise we will be
                in_position = hf.in_position(api)

                # Running wikapedia scraper to populate initial sp500 csv with ticker symbols of sp500 companies
                setup.create_crypto_csv(location_of_main_csv_file) #CHANGE THIS BACK FOR SP500

                # Setting our stockdf to be the csv file we just created
                og_df = pd.read_csv(location_of_main_csv_file, index_col='Symbol')

                # Creating our tickers list which we will pass into the Calculate_features function
                tickers = setup.Create_list_of_tickers(og_df.index)

                # Calculating our features for each ticker and populating the dataframe with these values
                setup.Calculate_features(tickers, og_df, crypto=crypto)

                # Dropping any columns that may have resulted in NaN values
                og_df = og_df.dropna()

                # Creating a scaled_data numpy array that we will pass into our K means algorithm
                scaled_data = setup.Scale_data(og_df)

                # Running k means
                setup.Apply_K_means(og_df, scaled_data, num_clusters)

                # Soring the data frame based on cluster value and saving these values to the dataframe, and then updating the csv
                setup.Sort_and_save(og_df, location_of_main_csv_file)

                # Creating a new dataframe that will contain information about the clusters that will be used for trading logic
                optimal_portfolio_allocation_info_df = setup.cluster_df_setup(hf.get_total_account_value(api), og_df)

                print("---------------------OPTIMAL PORTFOLIO ALLOCATION BASED ON TOTAL CURRENT ACCOUNT VALUE---------------------") 
                print(optimal_portfolio_allocation_info_df)
                print("---------------------OPTIMAL PORTFOLIO ALLOCATION BASED ON TOTAL CURRENT ACCOUNT VALUE---------------------")

                # Save the newsly created cluster df to a csv file
                setup.Save_cluster_df(optimal_portfolio_allocation_info_df, location_of_cluster_csv_file)
                
                # # Plotting cluster data
                # setup.plot_clusters(stockdf)


                if not in_position:
                    print("We have no positions open, so we will now form our initial optimized portfolio")
                    # For every cluster
                    for index, row in optimal_portfolio_allocation_info_df.iterrows():
                        # Create a list of the tickers
                        stocks_for_this_cluster = optimal_portfolio_allocation_info_df.loc[index, "Tickers"]
                        print(f"Symbols for cluster {index} are {stocks_for_this_cluster}")
                        if crypto:
                            stocks_for_this_cluster = [ticker.replace("-", "/") for ticker in stocks_for_this_cluster]
                            print(f"Since crypto, new symbols for cluster {index} are {stocks_for_this_cluster}")
                        # Store how much we should buy of each stock
                        dollars_per_stock_for_cluster = optimal_portfolio_allocation_info_df.loc[index, "Amount Per Stock"]
                        print(f"Cluster {index} will have ${dollars_per_stock_for_cluster} per stock alloted to it")
                        # For every stock in this cluster
                        for stock in stocks_for_this_cluster:
                            # Buy the specified amount of that stock
                            hf.execute_trade("buy", dollars_per_stock_for_cluster, stock, api, notional=True, crypto=crypto)
                            
                    in_position = True
                    print("Setting in_position = True and sending email notification")
                    current_portfolio_df = setup.Get_current_portfolio_allocation(optimal_portfolio_allocation_info_df, api, crypto)
                    print("---------------------CURRENT PORTFOLIO ALLOCATION AFTER INITAL ALLOCATIONS---------------------")
                    print(current_portfolio_df)
                    #setup.send_email("Entered Initial Positions", " ")
                    print("---------------------CURRENT PORTFOLIO ALLOCATION AFTER INITIAL ALLOCATIONS---------------------")
                if in_position:
                    print("We have positions open, so we will retreive them and see if they are optimzied")
                    current_portfolio_df = setup.Get_current_portfolio_allocation(optimal_portfolio_allocation_info_df, api, crypto)
                    # Need to recreate an optimal portfolio based on our new account values
                    optimal_portfolio_allocation_info_df = setup.cluster_df_setup(hf.get_total_account_value(api), og_df)
                    print("---------------------CURRENT PORTFOLIO ALLOCATION---------------------")
                    print(current_portfolio_df)
                    print("---------------------CURRENT PORTFOLIO ALLOCATION---------------------")
                    #setup.send_email("Entered Initial Positions", " ")
                    if setup.Is_balanced(current_portfolio_df, api):
                        print("The portfolio is balanced, no need to rebalance")
                        #('Portfolio is Still Balanced', ' ')
                    else:
                        #setup.send_email('Portfolio Needs Balancing', ' ')
                        while not setup.Is_balanced(current_portfolio_df, api):
                            most_unop_cluster_and_pct = setup.Get_most_unoptimized_cluster(current_portfolio_df)
                            most_unop_cluster = most_unop_cluster_and_pct[0]
                            most_unop_pct = most_unop_cluster_and_pct[1]

                            most_op_cluster_and_pct = setup.Get_most_optimized_cluster(current_portfolio_df)
                            most_op_cluster = most_op_cluster_and_pct[0]
                            most_op_pct = most_op_cluster_and_pct[1]


                            ##################################### HIGHER OR LOWER POSSIBLE FOR BOTH TO BE UNDER ALLOCATED
                            ############## REMEMBER THIS BRUH
                            print(f"Portfolio is not balanced: Most unoptimal cluster: {most_unop_cluster} by {most_unop_pct}")

                            total_portfolio_value = hf.get_total_account_value(api)#sum([float(hf.get_market_value(api, ticker, crypto=True)) for ticker in current_portfolio_df["Tickers"]]) + api.get_account().buying_power
                            dollar_target_allocation = optimal_portfolio_allocation_info_df.loc[most_unop_cluster, "Percentage"] * total_portfolio_value
                            dollar_current_allocation = current_portfolio_df.loc[most_unop_cluster, 'Dollars In Cluster']#sum([float(hf.get_market_value(api, ticker, crypto=True)) for ticker in optimal_portfolio_allocation_info_df.loc[most_unop_cluster, "Tickers"]])
                            dollar_trade_amount = (dollar_target_allocation - dollar_current_allocation)

                            # if the most unoptimzed cluster's off pct is negative, we will sell the ticker with the highest market value
                            # in the cluster that is the most optimized and then buy the ticker with the lowest market value in the 
                            # cluster that is the most unoptimized
                            
                            # if positive, most unop cluster is underallocated; buys inside unop cluster
                            if dollar_trade_amount > 0:
                                # if not enough money, sell highest value ticker in most optimized cluster
                                if float(api.get_account().buying_power) < dollar_trade_amount:
                                    print("We have no cash left, need to sell something")
                                    highest_market_value = -float('inf')
                                    ticker_to_sell = None
                                    for ticker in optimal_portfolio_allocation_info_df.loc[most_op_cluster, "Tickers"]:
                                        market_value = float(hf.get_market_value(api, ticker, crypto=True))
                                        if market_value > highest_market_value:
                                            highest_market_value = market_value
                                            ticker_to_sell = ticker

                                    if ticker_to_sell:
                                        ticker_to_sell = ticker_to_sell.replace("-", "/")
                                        try:
                                            print('WE ARE HERE 1')
                                            hf.execute_trade("sell", round(highest_market_value*0.96, 2), ticker_to_sell, api, notional=True, crypto=crypto)
                                            tm.sleep(10)  # Adjust the sleep time as needed for your platform's settlement time
                                        except Exception as e:
                                            print(f"Error executing sell order: {e}") 
                                # have enough money to buy within unoptimized cluster
                                print("Have necessary cash, can buy to reoptimize")
                                # if float(api.get_account().buying_power) < dollar_trade_amount:
                                #     dollar_trade_amount = float(api.get_account().buying_power)
                                lowest_market_value = float('inf')
                                ticker_to_buy = None
                                for ticker in optimal_portfolio_allocation_info_df.loc[most_unop_cluster, "Tickers"]:
                                    market_value = float(hf.get_market_value(api, ticker, crypto=True))
                                    if market_value < lowest_market_value:
                                        lowest_market_value = market_value
                                        ticker_to_buy = ticker

                                # Execute the buy order
                                if ticker_to_buy:
                                    ticker_to_buy = ticker_to_buy.replace("-", "/")
                                    try:
                                        hf.execute_trade("buy", round(dollar_trade_amount, 2), ticker_to_buy, api, notional=True, crypto=crypto)
                                    except Exception as e:
                                        print(f"Error executing buy order: {e}")
                            # Otherwise, if the most unoptimzed cluster's off pct is positive, we will sell the ticker with the highest market value
                            # in the most unoptimized cluster and then buy the ticker with the lowest market value in the 
                            # cluster that is the least optimized
                            
                            # if negative, most unop cluster is overallocated; buys inside most optimized cluster
                            else:
                                if float(api.get_account().buying_power) < abs(dollar_trade_amount):
                                    print("We have no cash left, need to sell something")
                                    highest_market_value = -float('inf')
                                    ticker_to_sell = None
                                    for ticker in optimal_portfolio_allocation_info_df.loc[most_unop_cluster, "Tickers"]:
                                        market_value = float(hf.get_market_value(api, ticker, crypto=True))
                                        if market_value > highest_market_value:
                                            highest_market_value = market_value
                                            ticker_to_sell = ticker

                                    if ticker_to_sell:
                                        ticker_to_sell = ticker_to_sell.replace("-", "/")
                                        try:
                                            print(f'WE ARE HERE 2; dollar trade amount is {dollar_trade_amount} and highest market value is {highest_market_value}')
                                            hf.execute_trade("sell", round(highest_market_value*0.96, 2), ticker_to_sell, api, notional=True, crypto=crypto)
                                            tm.sleep(10)  # Adjust the sleep time as needed for your platform's settlement time
                                        except Exception as e:
                                            print(f"Error executing sell order: {e}") 

                                print("Have necessary cash, can buy to reoptimize")
                                # if float(api.get_account().buying_power) < dollar_trade_amount:
                                #     dollar_trade_amount = float(api.get_account().buying_power)
                                lowest_market_value = float('inf')
                                ticker_to_buy = None
                                for ticker in optimal_portfolio_allocation_info_df.loc[most_op_cluster, "Tickers"]:
                                    market_value = float(hf.get_market_value(api, ticker, crypto=True))
                                    if market_value < lowest_market_value:
                                        lowest_market_value = market_value
                                        ticker_to_buy = ticker

                                # Execute the buy order
                                if ticker_to_buy:
                                    ticker_to_buy = ticker_to_buy.replace("-", "/")
                                    try:
                                        hf.execute_trade("buy", abs(round(dollar_trade_amount,2)), ticker_to_buy, api, notional=True, crypto=crypto)
                                    except Exception as e:
                                        print(f"Error executing buy order: {e}")
        
                            current_portfolio_df = setup.Get_current_portfolio_allocation(optimal_portfolio_allocation_info_df, api, crypto)
                            print("---------------------NEW PORTFOLIO ALLOCATION---------------------")
                            print(current_portfolio_df)
                            print("---------------------NEW PORTFOLIO ALLOCATION---------------------")
                            print('Chillin for 5 seconds')# FOR DEBUGGING
                            tm.sleep(5) # Letting things settle in the broker before trying again

                        # If we have any money left after rebalancing, just buy ticker with the lowest market value in the most optimized cluster
                        # until we run out of money
                    while float(api.get_account().buying_power) > 10:
                        lowest_market_value = float('inf')
                        ticker_to_buy = None
                        for ticker in optimal_portfolio_allocation_info_df.loc[setup.Get_most_optimized_cluster(current_portfolio_df)[0], "Tickers"]:
                            market_value = float(hf.get_market_value(api, ticker, crypto=True))
                            if market_value < lowest_market_value:
                                lowest_market_value = market_value
                                ticker_to_buy = ticker

                        # Execute the buy order
                        if ticker_to_buy:
                            ticker_to_buy = ticker_to_buy.replace("-", "/")
                            try:
                                hf.execute_trade("buy", 10, ticker_to_buy, api, notional=True, crypto=crypto)
                            except Exception as e:
                                print(f"Error executing buy order: {e}")
                        tm.sleep(5) # Letting stuff settle in api before recalling
                        #setup.send_email('Portfolio Rebalancing Complete', ' ')
                        print("Rebalancing Complete")
                        current_portfolio_df = setup.Get_current_portfolio_allocation(optimal_portfolio_allocation_info_df, api, crypto)
                        print("---------------------FINAL PORTFOLIO ALLOCATION---------------------")
                        print(current_portfolio_df)
                        print("---------------------FINAL PORTFOLIO ALLOCATION---------------------") 

            seconds_left_till_next_reallocation = setup.calculate_seconds_till_next_reallocation(est, hour_to_trade, minute_to_trade)

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
    # market_value = hf.get_market_value(api, 'LTC-USD', crypto=True)
    main()