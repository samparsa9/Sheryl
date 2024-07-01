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

def most_unoptimized_clusters(optimal_portfolio_allocation_df, api):
    """
    This function will take in the cluster information dataframe and the api,
    it will then see if the portfolio is balanced correctly depending on the cluster allocations,
    if it is not, it will return the two clusters off by the most percentage
    """
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

    return ((Highest_unoptimal_allocation_cluster, Highest_unoptimal_allocation_pct), (Lowest_unoptimal_allocation_cluster, Lowest_unoptimal_allocation_pct))
        
        




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
    optimal_portfolio_allocation_info_df = setup.cluster_df_setup(1000000, stockdf)
    print(optimal_portfolio_allocation_info_df)

    # optimized_portfolio_check(optimal_portfolio_allocation_info_df, api)
    # # Plotting cluster data
    # setup.plot_clusters(stockdf)

    


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
                


        except Exception as e:
            print(f"Error in main loop: {e}")
            tm.sleep(60)  # Wait for 1 minute before retrying
    

if __name__ == "__main__":
    main()

    

    

    
