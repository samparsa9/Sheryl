import pandas as pd

def create_crypto_csv(output_path):
    crypto_df = pd.DataFrame({
        "Symbol": ["AAVE/USD","AVAX/USD","BAT/USD","BCH/USD","BTC/USD","CRV/USD","DOGE/USD","DOT/USD","ETH/USD","LINK/USD","LTC/USD","MKR/USD","SHIB/USD","SUSHI/USD","UNI/USD","USDC/USD","USDT/USD","XTZ/USD"]
    })
    crypto_df.to_csv(output_path, index=False)
