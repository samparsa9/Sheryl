import alpaca_trade_api as tradeapi
import numpy as np
import talib as ta
import config as conf

# Function to make predictions
def predict_next_close(model, data, seq_length=60, close_mean=0, close_std=1):
    # Select only numeric columns and ensure they match the training features
    numeric_data = data[['open', 'high', 'low', 'close', 'volume']].tail(seq_length)
    # print("Last 60 rows of data before prediction:")
    # print(numeric_data)  # Print last seq_length rows of numeric data

    if len(numeric_data) < seq_length:
        print("Not enough data to make a prediction")
        return None

    # Normalize data using the provided mean and standard deviation
    numeric_data_normalized = (numeric_data - numeric_data.mean()) / numeric_data.std()
    recent_data = numeric_data_normalized.values.astype(np.float32)  # Ensure the data type is float32
    recent_data = recent_data.reshape(1, seq_length, -1)  # Reshape to (1, 60, 5)
    # print(f"Shape of recent_data after reshape: {recent_data.shape}")

    prediction = model.predict(recent_data)
    predicted_next_close = prediction[0][0]
    
    # Inverse transform the prediction to get the actual value
    normalized_next_close = predicted_next_close * close_std + close_mean
    
    return normalized_next_close



# Initialize Alpaca API
api = tradeapi.REST(conf.api_key, conf.api_secret, conf.base_url, api_version='v2')


def fetch_data(symbol, timeframe, limit=1000):  # Increased limit to fetch more data points
    bars = api.get_crypto_bars(symbol, timeframe).df.tail(limit)
    return bars

def get_available_balance(asset='BTC/USD'):
    try:
        position = api.get_position(asset)
        return float(position.qty)
    except tradeapi.rest.APIError as e:
        if e.status_code == 404:
            return 0.0
        else:
            raise e

def calculate_indicators(df):
    df['upper_band'], df['middle_band'], df['lower_band'] = ta.BBANDS(df['close'], timeperiod=20)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).rolling(conf.vwap_period).sum() / df['volume'].rolling(conf.vwap_period).sum()
    return df

def execute_trade(order_type, amount):
    try:
        if order_type == 'buy':
            order = api.submit_order(
                symbol=conf.symbol,
                qty=amount,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
        elif order_type == 'sell':
            available_balance = get_available_balance(conf.symbol)
            if available_balance < amount:
                amount = available_balance
            order = api.submit_order(
                symbol=conf.symbol,
                qty=amount,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
        print(f"Successfully Executed {order_type} for ",amount, " Bitcoin") # order: {order} <- include if you want all the order data as well
    except Exception as e:
        print(f"Error executing {order_type} order: {e}")
