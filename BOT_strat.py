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
import BOT_setup as bs

load_dotenv()
api_key = os.getenv('api_key')
api_secret = os.getenv("api_secret")
base_url = os.getenv('base_url')

# Initialize Alpaca API
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')



def main():
    in_position = False
    
    portfolio_amt = 1000000

    df = pd.DataFrame({
        "Percentage": [0.2, 0.4, 0.2, 0.1, 0.1], #percentage of allocation for each cluster
        "Dollars In Cluster": [0] * 5,
        "Num Stocks": [0] * 5, #number of stocks in each cluster
        "Amount Per Stock": [0] * 5,
        "Tickers": [[], [], [], [], []] #list of tickers for each cluster
    })

    #set the portfolio amount for each cluster
    df["Dollars In Cluster"] = df["Percentage"] * portfolio_amt
    print(df.head())

    stock_df = pd.read_csv("Sheryl/sp500_companies.csv", index_col='Symbol')

    #figure out how many stocks are in each cluster, and fill tickers column
    for ticker, row in stock_df.iterrows():
        if(row["Cluster"] == 0):
            df.at[0, "Tickers"].append(ticker)
            df.at[0, "Num Stocks"] += 1 
        elif(row["Cluster"] == 1):
            df.at[1, "Tickers"].append(ticker)
            df.at[1, "Num Stocks"] += 1 
        elif(row["Cluster"] == 2):
            df.at[2, "Tickers"].append(ticker)
            df.at[2, "Num Stocks"] += 1 
        elif(row["Cluster"] == 3):
            df.at[3, "Tickers"].append(ticker)
            df.at[3, "Num Stocks"] += 1 
        elif(row["Cluster"] == 4):
            df.at[4, "Tickers"].append(ticker)
            df.at[4, "Num Stocks"] += 1 
            
    for index, row in df.iterrows():
        #index is cluster num
        df.at[index, "Amount Per Stock"] = df.at[index, "Dollars In Cluster"] / df.at[index, "Num Stocks"]


    print(df.head())

    
    
    while True:
        try:
            print("----------New Clustering and Allocation Iteration---------")
            if not in_position:
                #need to buy in proportionally to each cluster
                
                
                
            #print(df)


        except Exception as e:
            print(f"Error in main loop: {e}")
            tm.sleep(60)  # Wait for 1 minute before retrying
    

if __name__ == "__main__":
    main()

    

    

    
