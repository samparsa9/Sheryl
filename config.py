import tensorflow as tf

# Loading Model
model = tf.keras.models.load_model('Alpacabot/Models/lstm_bitcoin_model.keras')

# Your API keys
api_key = 'PKF9EK8D88TLGMAAE00Z'
api_secret = 'XyrTXAgmVOltWbT0FGO5hrzamEZfxgtvRl7zNcuc'
base_url = 'https://paper-api.alpaca.markets'  # Remove /v2 from the base URL

# Parameters
symbol = 'BTC/USD'  # Change to the cryptocurrency pair you want to use
timeframe = '1Min'  # 1-minute bars
vwap_period = 14
amount = 1    # quanitity to trade
