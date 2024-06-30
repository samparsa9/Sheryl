# Strat Source: https://quantpedia.com/strategies/intraday-seasonality-in-bitcoin/

# Strat Description: This strategy will buy Bitcoin at exactly 21:00 UTC and sell at exactly 23:00, profiting off the volatility
# generated by excess volume flowing into the security since most major internation exchanges are closed during that period

import sys
import os
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta, timezone
import time as tm
import sys
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('api_key')
api_secret = os.getenv("api_secret")


# Import your custom modules
import config as conf
import helperfuncs as hf

# Initialize Alpaca API
api = tradeapi.REST(api_key, api_secret, conf.base_url, api_version='v2')


def main():
    position = None
    while True:
        try:
            print("---------------------------------------------Start of Iteration !!!----------------------------------------------")
           
            if position is None and (datetime.now(timezone.utc).hour == 21 and datetime.now(timezone.utc).minute == 0):
                print("It is now 21:00 UTC time, ENTERING BITCOIN POSITION")
                hf.execute_trade('buy', conf.amount)
                position = 'long'
            else:
                pass

            if position == 'long' and (datetime.now(timezone.utc).hour == 23 and datetime.now(timezone.utc).minute == 0):
                print("It is now 23:00 UTC time, CLOSING BITCOIN POSITION")
                api.close_all_positions()
                position = None
            else:
                pass    
            
            def calculate_seconds_till_next_trade_window():
                now = datetime.now(timezone.utc)
                target_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
                if now > target_time:
                    target_time += timedelta(days=1)
                return int((target_time - now).total_seconds())
            
            seconds_left_till_next_trade_window = calculate_seconds_till_next_trade_window()

            for seconds_left in range(seconds_left_till_next_trade_window, 0, -1):
                hours_left = seconds_left // 3600
                minutes_left = (seconds_left % 3600) // 60
                remaining_seconds = seconds_left % 60
                sys.stdout.write(f"\r{hours_left} hours, {minutes_left} minutes, {remaining_seconds} seconds till next iteration!!!")
                sys.stdout.flush()
                tm.sleep(1)
            print("It's 21:00 UTC, time for the next trade window!")
        except Exception as e:
            print(f"Error in main loop: {e}")
            tm.sleep(60)  # Wait for 1 minute before retrying


if __name__ == "__main__":
    main()
