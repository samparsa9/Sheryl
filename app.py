from flask import Flask, render_template, jsonify
import boto3
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

aws_key = os.getenv('AWS_KEY')
aws_secret = os.getenv('AWS_SECRET')

app = Flask(__name__)

# Initialize DynamoDB resource
dynamodb = boto3.resource('dynamodb', region_name='us-east-1', 
                          aws_access_key_id=aws_key,
                          aws_secret_access_key=aws_secret)

# Define DynamoDB tables
portfolio_overview_table = dynamodb.Table('portfolio_overview')
portfolio_positions_table = dynamodb.Table('portfolio_positions')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/portfolio-overview')
def get_portfolio_overview():
    response = portfolio_overview_table.scan()
    items = response.get('Items', [])
    return jsonify(items)

@app.route('/portfolio-positions/<overview_id>')
def get_portfolio_positions(overview_id):
    response = portfolio_positions_table.scan(
        FilterExpression=boto3.dynamodb.conditions.Attr('overview_id').eq(overview_id)
    )
    items = response.get('Items', [])
    return jsonify(items)

if __name__ == '__main__':
    app.run(debug=True)
