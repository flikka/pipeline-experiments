from azure.storage.blob import BlockBlobService
import os
import pandas
import pyodbc

ACCOUNT_NAME = os.environ['ACCOUNT_NAME']
ACCOUNT_KEY = os.environ['BLOB_KEY']
CONTAINER_NAME = os.environ['CONTAINER_NAME']
SQL_SERVER = os.environ['SQL_SERVER']
DATABASE = os.environ['DATABASE']
DB_USER = os.environ['DB_USER']
DB_PW = os.environ['DB_PW']

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

def load_from_azure_sql():
    
    driver= '{ODBC Driver 13 for SQL Server}'
    connection = pyodbc.connect('DRIVER='+ driver +';PORT=1433;SERVER='+SQL_SERVER+'\
        ;PORT=1443;DATABASE='+DATABASE+';UID='+DB_USER+';PWD='+ DB_PW)
    
    print("Downloading data from database {} on server {}".format(DATABASE, SQL_SERVER))
    dataframe = pandas.read_sql("SELECT * from diamonds", connection) 
    return dataframe
    
def wipe_blob_container(container_name):
    blob_service = BlockBlobService(account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)
    generator = blob_service.list_blobs(container_name)
    for blob in generator:
        if blob.name.startswith("results"):
            print("Deleting blob: {}".format(blob.name))
            blob_service.delete_blob(CONTAINER_NAME, blob.name)
    
if __name__=="__main__":
    pass
    #print(download_input("diamonds.csv"))
    #load_from_azure_sql()
    #wipe_blob_container(CONTAINER_NAME)
    