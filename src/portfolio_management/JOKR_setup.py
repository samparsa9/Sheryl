import pandas as pd
import os
import src.utils.alpaca_utils as hf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from datetime import datetime, timedelta


load_dotenv()
# Email Feature info
sender = os.getenv('sender')
recipient = os.getenv('sender')
password = os.getenv('email_password')


def send_email(subject, message):

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject

    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        text = msg.as_string()
        server.sendmail(sender, recipient, text)
        server.quit()
        # print('Email sent')
    except Exception as e:
        print(f"Failed to send email: {e}")


def cluster_df_setup(starting_cash, stock_df):
    portfolio_amt = starting_cash

    # CHANGE ALL THIS BACK
    cluster_info_df = pd.DataFrame({
             #"Cluster": [0,1,2,3,4],
            "Cluster": [0,1,2,3],
            #"Percentage": [0.2, 0.4, 0.2, 0.1, 0.1], #percentage of allocation for each cluster
            "Percentage": [0.2, 0.4, 0.2, 0.2],
            #"Dollars In Cluster": [0.0] * 5.0,
            "Dollars In Cluster": [0.0] * 4,
            #"Num Stocks": [0.0] * 5, #number of stocks in each cluster
            "Num Stocks": [0.0] * 4,
            #"Amount Per Stock": [0.0] * 5,
            "Amount Per Stock": [0.0] * 4,
            #"Tickers": [[], [], [], [], []] #list of tickers for each cluster
            "Tickers": [[], [], [], []]
        })

    # Set the portfolio amount for each cluster
    cluster_info_df["Dollars In Cluster"] = round((cluster_info_df["Percentage"] * portfolio_amt),2)
    cluster_info_df.set_index("Cluster", inplace=True)
    #figure out how many stocks are in each cluster, and fill tickers column
    for ticker, row in stock_df.iterrows():
        # print("----------------------------------------------------------------------------")
        # print(ticker)
        # print(row)
        # print("----------------------------------------------------------------------------")
        cluster = row["Cluster"]
        # print(stock_df.head())
        # print(cluster_info_df)
        cluster_info_df.loc[cluster, "Tickers"].append(ticker)
        cluster_info_df.loc[cluster, "Num Stocks"] += 1

    for index, row in cluster_info_df.iterrows():
        # index is the cluster number
        if cluster_info_df.loc[index, "Num Stocks"] > 0:
            cluster_info_df.loc[index, "Amount Per Stock"] = round((cluster_info_df.loc[index, "Dollars In Cluster"] / cluster_info_df.loc[index, "Num Stocks"]), 2)
        else:
            cluster_info_df.loc[index, "Amount Per Stock"] = 0  # Avoid division by zero
    
    return cluster_info_df

def Get_current_portfolio_allocation(optimal_portfolio_allocation_df, api, crypto=False):
    # Create a new dataframe that represents our current portfolios dollar allocation each clusters
    current_portfolio_allocation = pd.DataFrame({
            #"Cluster": [0,1,2,3,4],
            "Cluster": [0,1,2,3],
            "Current Pct Allocation": [0.0] * 4,
            "Pct Off From Optimal": [0.0] * 4,
            #"Dollars In Cluster": [0] * 5,
            "Dollars In Cluster": [0.0] * 4
        })
    current_portfolio_allocation.set_index("Cluster", inplace=True)
    # This for loop will be used to calulate the total dollars in each cluster by looping through each ticker in each cluster
    # Snd adding the market value of our position in that ticker to a running sum
    for cluster in optimal_portfolio_allocation_df.index:
        dollars_in_this_cluster = 0.0
        tickers = optimal_portfolio_allocation_df.loc[cluster, "Tickers"]
        for ticker in tickers:
            #print("here is the ticker before its sent to the func: " + ticker)
            market_value = hf.get_market_value(ticker, crypto)
            # print(f"Ticker: {ticker}, Market Value: {market_value}")
            dollars_in_this_cluster += market_value
        # Populating the Dollars In Cluster column with these new values in our current portfolio allocation df
        current_portfolio_allocation.loc[cluster, "Dollars In Cluster"] = round(dollars_in_this_cluster, 2)
        # print(f"Cluster {cluster}, Dollars In Cluster: {dollars_in_this_cluster}")
    for cluster in optimal_portfolio_allocation_df.index:
        dollars_in_this_cluster = current_portfolio_allocation.loc[cluster, 'Dollars In Cluster']
        account_value = hf.get_total_account_value()

        current_cluster_pct_allocation = (dollars_in_this_cluster / account_value)
        optimal_cluster_pct_allocation = optimal_portfolio_allocation_df.loc[cluster, "Percentage"]

        current_portfolio_allocation.loc[cluster, "Current Pct Allocation"] = current_cluster_pct_allocation
        current_portfolio_allocation.loc[cluster, "Pct Off From Optimal"] = current_cluster_pct_allocation - optimal_cluster_pct_allocation

    return current_portfolio_allocation


def Is_balanced(current_portfolio_df):
    largest_pct_off_by = 0
    for cluster, row in current_portfolio_df.iterrows():
        this_cluster_off_by = current_portfolio_df.loc[cluster, "Pct Off From Optimal"]
        if abs(this_cluster_off_by) > abs(largest_pct_off_by):
            largest_pct_off_by = this_cluster_off_by
    return abs(largest_pct_off_by) < 0.03 #changed threshold


def calculate_seconds_till_next_reallocation(timezone, hour_to_trade, minute_to_trade):
                now = datetime.now(timezone)
                target_time = now.replace(hour=hour_to_trade, minute=minute_to_trade, second=0, microsecond=0)
                if now > target_time:
                    target_time += timedelta(days=1)
                return int((target_time - now).total_seconds())


def main():
    pass
if __name__ == "__main__":
    main()
