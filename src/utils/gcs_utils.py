from google.cloud import storage
import os
from dotenv import load_dotenv
from config.settings import GOOGLE_APPLICATION_CREDENTIALS_PATH

load_dotenv()

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS_PATH
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)