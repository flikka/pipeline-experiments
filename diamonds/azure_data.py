from azure.storage.blob import BlockBlobService
import os
#import pyodbc
ACCOUNT_NAME = "dataplatformkflikstorage"
ACCOUNT_KEY = os.environ['BLOB_KEY']
CONTAINER_NAME = "diamonds"


def download_input_blob(filename):
    blob_service = BlockBlobService(account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)
    print("Downloading file <{}> from {} on account {}".format(filename, CONTAINER_NAME, ACCOUNT_NAME))
    return blob_service.get_blob_to_bytes(CONTAINER_NAME, filename)


def upload_text_to_blob(text, blob_name):
    blob_service = BlockBlobService(account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)
    print("Uploading to {} in Azure".format(str(blob_name)))
    blob_service.create_blob_from_text(CONTAINER_NAME, str(blob_name), text)
    
def upload_bytes_to_blob(bytesIO, blob_name):
    blob_service = BlockBlobService(account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)
    print("Uploading to {} in Azure".format(str(blob_name)))
    blob_service.create_blob_from_stream(CONTAINER_NAME, str(blob_name), bytesIO)
