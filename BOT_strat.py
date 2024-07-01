import BOT_setup as bs
import sys
import os
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta, timezone
import time as tm
import sys
from dotenv import load_dotenv
import Alpacahelperfuncs as hf
import pandas as pd
import BOT_setup as setup

load_dotenv()
api_key = os.getenv('api_key')
api_secret = os.getenv("api_secret")
base_url = os.getenv('base_url')

# Initialize Alpaca API
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')


def main():
    # When we run this algo for the very first time we won't be in a position
    in_position = False
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
    cluster_df = setup.cluster_df_setup(1000000, stockdf)
    # Plotting cluster data
    setup.plot_clusters(stockdf)
    


    while not True: # FOR DEBUGGING, SET BACK TO TRUE
        try:
            print("----------New Clustering and Allocation Iteration---------")
            if not in_position and (datetime.now(timezone.utc).hour == 14 and datetime.now(timezone.utc).minute == 00):
                #need to buy in proportionally to each cluster
                for index, row in cluster_df.iterrows():
                    stocks_for_this_cluster = cluster_df.at[index, "Tickers"]
                    dollars_per_stock_for_cluster = cluster_df.at[index, "Amount Per Stock"]
                    for stock in stocks_for_this_cluster:
                        hf.execute_trade("buy", dollars_per_stock_for_cluster, stock, api)
            if in_position and (datetime.now(timezone.utc).hour == 14 and datetime.now(timezone.utc).minute == 00):
                return 0
                
            #print(df)


        except Exception as e:
            print(f"Error in main loop: {e}")
            tm.sleep(60)  # Wait for 1 minute before retrying
    

if __name__ == "__main__":
    main()

    

    

    
