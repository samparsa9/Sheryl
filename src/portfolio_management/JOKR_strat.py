import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
import time as tm
import sys
from utils import alpaca_utils as hf
import pandas as pd
import pytz
import traceback
import JOKR_setup as setup
import sys
import os

# Add the parent directory of src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import configuration settings
from config.settings import CSV_DIRECTORY, FRED_KEY, GCS_BUCKET_NAME

# Import utility functions
from utils.gcs_utils import upload_to_gcs

# Import data collection functions
from data_collection.calculate_features import calculate_features
from data_collection.crypto_scraper import create_crypto_csv
from data_collection.sp500_scraper import create_sp500_csv

# Import data processing functions
from data_processing.scaler import scale_data
from data_processing.k_means_cluster import apply_k_means




def main():
    ############ STATIC STUFF AMONG ITERATIONS ################
    ############ ONLY BEING RUN FIRST TIME WE EVER LAUNCH ALGO #############


    while True: # initially collecting data to form clusters and the initial optimal portfolio balance
        # Get the current time in EST
        est = pytz.timezone('US/Eastern')
        now = datetime.now(est)

        # The Hour and Minute at which we want our bot to check allocations
        hour_to_trade = int(now.hour) # SET BACK
        minute_to_trade = int(now.minute) # SET BACK

        os.makedirs(CSV_DIRECTORY, exist_ok=True)

        # Set true since were dealing with crypto symbols
        crypto = True # SET BACK TO FALSE

        # SET BACK TO 5
        num_clusters = 4 # Use this to specify how many cluster for K-means

        if crypto:
            create_crypto_csv(os.path.join(CSV_DIRECTORY, 'crypto_df.csv'))
            feature_df = pd.read_csv(os.path.join(CSV_DIRECTORY, 'crypto_df.csv'), index_col='Symbol')
        else:
            create_sp500_csv(os.path.join(CSV_DIRECTORY, 'sp500_df.csv'))
            feature_df = pd.read_csv(os.path.join(CSV_DIRECTORY, 'sp500_df.csv'), index_col='Symbol')

        symbols = feature_df.index.tolist()
        calculate_features(symbols, feature_df, FRED_KEY, crypto=crypto)

        feature_df.dropna(inplace=True)
        features = feature_df.columns.tolist()
        scaled_data = scale_data(feature_df, features)
        apply_k_means(feature_df, scaled_data, num_clusters=4)
        feature_df = feature_df.sort_values(by='Cluster')
        feature_df.to_csv(os.path.join(CSV_DIRECTORY, 'processed_data.csv'))

        #upload_to_gcs(GCS_BUCKET_NAME, os.path.join(CSV_DIRECTORY, 'processed_data.csv'), 'Data/processed_data.csv')

        #send_email(SENDER_EMAIL, SENDER_EMAIL, 'ETL Process Complete', 'The ETL process has been successfully completed.', EMAIL_PASSWORD)

        try:
            print("-------------------New Clustering and Allocation Iteration------------------")

            # If the time is 10:00 AM, set up all data that will be needed for allocation/reallocation
            if (now.hour == hour_to_trade and now.minute == minute_to_trade): # CHANGE THE TIME TO DEBUG
                # When we run this algo for the very first time we won't be in a position, otherwise we will be
                in_position = hf.in_position()

                # Creating a new dataframe that will contain information about the clusters that will be used for trading logic
                master_df = hf.cluster_and_allocation_setup(hf.get_total_account_value(),feature_df, num_clusters=num_clusters, crypto=crypto )

                print("---------------------OPTIMAL PORTFOLIO ALLOCATION BASED ON TOTAL CURRENT ACCOUNT VALUE---------------------") 
                print(master_df)
                print("-----------------------------------------------------------------------------------------------------------")

                master_df= master_df.sort_values(by='Cluster')
                master_df.to_csv(os.path.join(CSV_DIRECTORY, 'optimal_vs_current portfolio.csv'))

                #upload_to_gcs(GCS_BUCKET_NAME, os.path.join(CSV_DIRECTORY, 'optimal_portfolio_info'), 'Data/optimal_portfolio_info')

                if not in_position:
                    print("We have no positions open, so we will now form our initial optimized portfolio")
                    # For every cluster
                    for index, row in master_df.iterrows():
                        # Create a list of the tickers
                        print('-'*50)
                        stocks_for_this_cluster = master_df.loc[index, "Tickers"]
                        print(f"Symbols for cluster {index} are {stocks_for_this_cluster}")
                        if crypto:
                            stocks_for_this_cluster = [ticker.replace("-", "/") for ticker in stocks_for_this_cluster]
                            print(f"Since crypto, new symbols for cluster {index} are {stocks_for_this_cluster}")
                        # Store how much we should buy of each stock
                        dollars_per_stock_for_cluster = master_df.loc[index, "$ Per Stock"]
                        print(f"Cluster {index} will have ${dollars_per_stock_for_cluster} per stock alloted to it")
                        print('-'*50)
                        # For every stock in this cluster
                        for stock in stocks_for_this_cluster:
                            # Buy the specified amount of that stock
                            hf.execute_trade("buy", dollars_per_stock_for_cluster, stock, notional=True, crypto=crypto)
                            tm.sleep(1)
                            
                    in_position = True
                    master_df = hf.cluster_and_allocation_setup(hf.get_total_account_value(), feature_df, num_clusters, crypto)
                    print("---------------------INITIAL POSITION AFTER NOT BEING IN POSITIONS---------------------")
                    print(master_df)
                    #setup.send_email("Entered Initial Positions", " ")
                    print("---------------------------------------------------------------------------------------")
                
                if in_position:
                    print("We have positions open, so we will retreive them and see if they are optimzed")
                    master_df = hf.cluster_and_allocation_setup(hf.get_total_account_value(), feature_df, num_clusters, crypto)
                    # Need to recreate an optimal portfolio based on our new account values
                    print("---------------------CURRENT PORTFOLIO ALLOCATION---------------------")
                    print(master_df)
                    print("----------------------------------------------------------------------")
                    #setup.send_email("Entered Initial Positions", " ")
                    if setup.Is_balanced(master_df):
                        print("The portfolio is balanced, no need to rebalance")
                        #('Portfolio is Still Balanced', ' ')
                    else:
                        # Step 1: Go through each cluster, look for overallocated clusters. If found, sell from tickers to reach optimized percentage.
                        print('---------------------SELLING OVERALLOCATED CLUSTERS---------------------')
                        overallocated_df = master_df[master_df['Pct Off Op'] > 0]
                        for cluster, row in overallocated_df.iterrows():
                            while master_df.loc[cluster, 'Pct Off Op'] > 0.01:
                                highest_market_value = -float('inf')
                                ticker_to_sell = None
                                for ticker in master_df.loc[cluster, "Tickers"]:
                                    market_value = float(hf.get_market_value(ticker, crypto=True))
                                    if market_value > highest_market_value:
                                        highest_market_value = market_value
                                        ticker_to_sell = ticker
                                if ticker_to_sell:
                                    ticker_to_sell = ticker_to_sell.replace("-", "")
                                    ticker_to_sell = ticker_to_sell.replace("/", "")
                                    try:
                                        amount_to_sell = 100 #hf.get_available_balance(api, ticker_to_sell)
                                        hf.execute_trade("sell", amount_to_sell, ticker_to_sell, notional=True, crypto=crypto)
                                        tm.sleep(2)  # Adjust the sleep time as needed for your platform's settlement time
                                        # print("---------------------NEW PORTFOLIO ALLOCATION---------------------")
                                        # print(current_portfolio_df)
                                        # print("---------------------NEW PORTFOLIO ALLOCATION---------------------")
                                    except Exception as e:
                                        print(f"Error executing sell order: {e}") 
                                master_df = hf.cluster_and_allocation_setup(hf.get_total_account_value(), feature_df, num_clusters, crypto)
                        print("-"*50)
                        # Step 2: Go through each cluster, look for underallocated clusters. If found, buy tickers in cluster to reach optimized percentage.
                        print('---------------------BUYING UNDERALLOCATED CLUSTERS---------------------')
                        underallocated_df = master_df[master_df['Pct Off Op'] < 0]
                        for cluster, row in underallocated_df.iterrows():
                            ############################################################################
                            while abs(master_df.loc[cluster, 'Pct Off Op']) > 0.03: # CHANGED THIS NUMBER FROM 0.03 TO 0.015 AND WORKS FLAWLESSELY, COULD MAYBE LEAD TO ERROR
                                # Did BARELY lead to an error, upped it a tiny bit
                            ############################################################################
                                lowest_market_value = float('inf')
                                ticker_to_buy = None
                                for ticker in master_df.loc[cluster, "Tickers"]:
                                    market_value = float(hf.get_market_value(ticker, crypto=True))
                                    if market_value < lowest_market_value:
                                        lowest_market_value = market_value
                                        ticker_to_buy = ticker
                                # Execute the buy order
                                if ticker_to_buy:
                                    ticker_to_buy = ticker_to_buy.replace("-", "")
                                    ticker_to_buy = ticker_to_buy.replace("/", "")
                                    try:
                                        # print(f"trying to buy {dollar_trade_amount} worth of {ticker_to_buy}")
                                        # print(f'Remaining buying power: {api.get_account().buying_power}')
                                        amount_to_buy = 100
                                        hf.execute_trade("buy", amount_to_buy, ticker_to_buy, notional=True, crypto=crypto)
                                        tm.sleep(2)
                                        # print(f'Remaining buying power after purchase: {api.get_account().buying_power}')
                                        # print("---------------------NEW PORTFOLIO ALLOCATION---------------------")
                                        # print(current_portfolio_df)
                                        # print("---------------------NEW PORTFOLIO ALLOCATION---------------------")
                                    except Exception as e:
                                        print(f"Error executing buy order: {e}")
                                master_df = hf.cluster_and_allocation_setup(hf.get_total_account_value(), feature_df, num_clusters, crypto)
                        print("-"*50)

                        # Step 3: Divide remaining buying power by # of clusters and distribute equally into them
                        if float(hf.get_buying_power()) > 15:
                            print('---------------------USING EXCESS CASH TO DISTRUBUTE EVENLY---------------------')
                            total_amount_to_distribute = float(hf.get_buying_power)
                            individual_amount_to_distribute = float(total_amount_to_distribute / num_clusters)
                            for cluster, row in master_df.iterrows():
                                lowest_market_value = float('inf')
                                ticker_to_buy = None
                                for ticker in master_df.loc[cluster, "Tickers"]:
                                    market_value = float(hf.get_market_value(ticker, crypto=True))
                                    if market_value < lowest_market_value:
                                        lowest_market_value = market_value
                                        ticker_to_buy = ticker
                                # Execute the buy order
                                if ticker_to_buy:
                                    ticker_to_buy = ticker_to_buy.replace("-", "")
                                    ticker_to_buy = ticker_to_buy.replace("/", "")
                                    try:
                                        # print(f"trying to buy {dollar_trade_amount} worth of {ticker_to_buy}")
                                        amount_to_buy = round(individual_amount_to_distribute,2)
                                        hf.execute_trade("buy", amount_to_buy, ticker_to_buy, notional=True, crypto=crypto)
                                        tm.sleep(2)
                                    except Exception as e:
                                        print(f"Error executing buy order: {e}")
                        print('-'*50)
                        #setup.send_email('Portfolio Rebalancing Complete', ' ')
                        print("***REBALANCING COMPLETE***")
                        master_df = hf.cluster_and_allocation_setup(hf.get_total_account_value(), feature_df, num_clusters, crypto)
                        print("---------------------FINAL PORTFOLIO ALLOCATION---------------------")
                        print(master_df)
                        print("--------------------------------------------------------------------") 

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