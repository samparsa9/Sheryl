from src.utils.alpaca_utils import get_positions, get_total_account_value
from src.utils.gcs_utils import upload_to_gcs
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String, ForeignKey, TIMESTAMP
from datetime import datetime
import pandas as pd
import pytz
import schedule
import time
from config import settings

DATABASE_URI = 'postgresql+psycopg2://postgres:Shery1J0kerBot456@34.150.254.10:5432/financial_data'
engine = create_engine(DATABASE_URI)

# Get the current time in EST
est = pytz.timezone('US/Eastern')

# Define metadata
metadata = MetaData()

# Define tables
portfolio_overview = Table('portfolio_overview', metadata,
                           Column('id', Integer, primary_key=True),
                           Column('timestamp', TIMESTAMP),
                           Column('total_account_value', Float))

portfolio_positions = Table('portfolio_positions', metadata,
                            Column('id', Integer, primary_key=True),
                            Column('overview_id', Integer, ForeignKey('portfolio_overview.id')),
                            Column('symbol', String),
                            Column('quantity', Float),
                            Column('market_value', Float))

# Create tables if they do not exist
metadata.create_all(engine)

def upload_portfolio_metrics_to_db():
    try:
        # Fetch total account value
        total_account_value = get_total_account_value()

        # Insert into portfolio_overview table
        conn = engine.connect()
        result = conn.execute(portfolio_overview.insert().values(
            timestamp=datetime.now(est),
            total_account_value=total_account_value
        ))
        overview_id = result.inserted_primary_key[0]

        # Fetch current positions
        positions = get_positions()

        # Prepare data for insertion
        position_data = []
        for position in positions:
            data = {
                'overview_id': overview_id,
                'symbol': position.symbol,
                'quantity': float(position.qty),
                'market_value': float(position.market_value)
            }
            position_data.append(data)

        # Insert into portfolio_positions table
        conn.execute(portfolio_positions.insert(), position_data)
        print(f"Uploaded portfolio metrics at {datetime.utcnow()}")
        conn.close()

    except Exception as e:
        print(f"Error fetching or uploading portfolio metrics: {e}")

# Schedule the function to run every 5 minutes
schedule.every(1).minutes.do(upload_portfolio_metrics_to_db)

# Run the schedule
while True:
    schedule.run_pending()
    time.sleep(1)

