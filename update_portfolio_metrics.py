from src.utils.alpaca_utils import get_positions, get_total_account_value
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, ForeignKey, MetaData, TIMESTAMP
from sqlalchemy.sql import insert
from datetime import datetime
import pytz
import schedule
import time
from config.settings import DATABASE_URI

# Initialize the SQLAlchemy engine
engine = create_engine(DATABASE_URI)

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

# Get the current time in EST
est = pytz.timezone('US/Eastern')

def upload_portfolio_metrics_to_db():
    try:
        # Fetch total account value
        total_account_value = get_total_account_value()
        print(f"Total Account Value: {total_account_value}")

        # Insert into portfolio_overview table
        with engine.begin() as conn:  # Use begin() to ensure the transaction is committed
            result = conn.execute(insert(portfolio_overview).values(
                timestamp=datetime.now(est),
                total_account_value=total_account_value
            ))
            overview_id = result.inserted_primary_key[0]
            print(f"Inserted portfolio overview with ID: {overview_id}")

            # Fetch current positions
            positions = get_positions()
            print(f"Fetched {len(positions)} positions")

            # Prepare data for insertion
            position_data = [
                {
                    'overview_id': overview_id,
                    'symbol': position.symbol,
                    'quantity': float(position.qty),
                    'market_value': float(position.market_value)
                }
                for position in positions
            ]

            print(f"Position Data: {position_data}")

            # Insert into portfolio_positions table using execute for batch insertion
            conn.execute(portfolio_positions.insert(), position_data)
            print(f"Inserted {len(position_data)} positions")

        print(f"Uploaded portfolio metrics at {datetime.utcnow()}")

    except Exception as e:
        print(f"Error fetching or uploading portfolio metrics: {e}")


# Run the schedule
while True:
    upload_portfolio_metrics_to_db()
    time.sleep(1)
