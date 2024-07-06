import requests
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import time
from config.settings import DATABASE_URI

engine = create_engine(DATABASE_URI)

def fetch_bitcoin_price():
    url = 'https://api.coinbase.com/v2/prices/BTC-USD/spot'
    response = requests.get(url)
    data = response.json()
    return {
        'timestamp': datetime.now(),
        'price': float(data['data']['amount'])
    }

def store_data(data):
    df = pd.DataFrame([data])
    df.to_sql('BitcoinMinuteData', engine, if_exists='append', index=False)

print(1)
def main():
    while True:
        try:
            print(2)
            data = fetch_bitcoin_price()
            print(3)
            store_data(data)
            print(4)
            print(f"Recorded data at {data['timestamp']}: ${data['price']}")
        except Exception as e:
            print(f"Error: {e}")
        # Wait for 60 seconds
        time.sleep(60)

if __name__ == '__main__':
    main()
