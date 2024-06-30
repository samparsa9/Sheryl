import tensorflow as tf

# Loading Model
model = tf.keras.models.load_model('Alpacabot/Models/lstm_bitcoin_model.keras')

# base url
base_url = 'https://paper-api.alpaca.markets'  # Remove /v2 from the base URL

# Parameters
symbol = 'BTC/USD'  # Change to the cryptocurrency pair you want to use
timeframe = '1Min'  # 1-minute bars
vwap_period = 14
amount = 1    # quanitity to trade
