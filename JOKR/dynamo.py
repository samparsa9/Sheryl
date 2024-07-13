import boto3
from datetime import datetime
import pytz
import time
from decimal import Decimal
from src.utils.alpaca_utils import get_positions, get_total_account_value
import os
import dotenv

aws_key = os.getenv('AWS_KEY')
aws_secret = os.getenv('AWS_SECRET')

# Initialize DynamoDB resource
dynamodb = boto3.resource('dynamodb', region_name='us-east-1', 
                          aws_access_key_id=aws_key,
                          aws_secret_access_key=aws_secret)

# Define DynamoDB tables
portfolio_overview_table = dynamodb.Table('portfolio_overview')
portfolio_positions_table = dynamodb.Table('portfolio_positions')

# Get the current time in EST
est = pytz.timezone('US/Eastern')

def convert_to_str(data):
    """Convert float values to string to avoid precision issues with DynamoDB."""
    if isinstance(data, dict):
        return {k: convert_to_str(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_str(i) for i in data]
    elif isinstance(data, float):
        return str(data)
    return data

def upload_portfolio_metrics_to_dynamodb():
    try:
        # Fetch total account value
        total_account_value = get_total_account_value()
        print(f"Total Account Value: {total_account_value}")

        # Insert into portfolio_overview table
        timestamp = datetime.now(est).isoformat()
        overview_id = str(datetime.now().timestamp())
        overview_item = {
            'id': overview_id,  # unique ID based on timestamp
            'timestamp': timestamp,
            'total_account_value': str(total_account_value)  # Convert to string
        }
        portfolio_overview_table.put_item(Item=overview_item)
        print(f"Inserted portfolio overview with ID: {overview_id}")

        # Fetch current positions
        positions = get_positions()
        print(f"Fetched {len(positions)} positions")

        # Prepare data for insertion
        position_data = [
            {
                'overview_id': overview_id,  # Foreign key reference
                'position_id': f"{overview_id}-{position.symbol}-{i}",  # Unique position ID
                'symbol': position.symbol,
                'quantity': str(position.qty),  # Convert to string
                'market_value': str(position.market_value)  # Convert to string
            }
            for i, position in enumerate(positions)
        ]

        print(f"Position Data: {position_data}")

        # Insert into portfolio_positions table
        with portfolio_positions_table.batch_writer() as batch:
            for item in position_data:
                batch.put_item(Item=item)

        print(f"Inserted {len(position_data)} positions")

        print(f"Uploaded portfolio metrics at {datetime.utcnow()}")

    except Exception as e:
        print(f"Error fetching or uploading portfolio metrics: {e}")

# Run the schedule
while True:
    upload_portfolio_metrics_to_dynamodb()
    time.sleep(1)
