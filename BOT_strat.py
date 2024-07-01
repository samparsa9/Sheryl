import BOT_setup as bs
import sys
import os
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta, timezone
import time as tm
import sys
from dotenv import load_dotenv
import Sheryl.Alpacahelperfuncs as hf
import pandas as pd
import BOT_setup as setup

load_dotenv()
api_key = os.getenv('api_key')
api_secret = os.getenv("api_secret")
base_url = os.getenv('base_url')

# Initialize Alpaca API
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')


def main():
    in_position = False

    stockdf = pd.read_csv('sp500_companies.csv', index_col='Symbol')
    # stockdf = stockdf.drop(columns=["Security", "GICS Sector", "GICS Sub-Industry", "Headquarters Location", "Date added", "CIK", "Founded"])
    stockdf = stockdf.drop(columns=["Cluster"])
    stockdf = stockdf.dropna()
    tickers = setup.Create_list_of_tickers(stockdf.index)
    setup.Calculate_features(tickers, stockdf)
    scaled_data = setup.Scale_data(stockdf)
    setup.Apply_K_means(stockdf, scaled_data)
    setup.Sort_and_save(stockdf)
    cluster_df = setup.cluster_df_setup(1000000, stockdf)
    


    while True:
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

    

    

    
