import os
from dotenv import load_dotenv

load_dotenv()
# Getting all of our credentials from our local .env file
API_KEY = os.getenv('api_key')
API_SECRET = os.getenv('api_secret')
BASE_URL = os.getenv('base_url')
FRED_KEY = os.getenv('fred_key')
CSV_DIRECTORY = os.getenv('DATA_directory')
SENDER_EMAIL = os.getenv('sender')
EMAIL_PASSWORD = os.getenv('email_password')
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
GOOGLE_APPLICATION_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_PATH')
DATABASE_URI = os.getenv('DATABASE_URI')
AWS_KEY = os.getenv('AWS_KEY')
AWS_SECRET = os.getenv('AWS_SECRET')