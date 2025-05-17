import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv
import logging


logger = logging.getLogger(__name__)
logger.level = logging.INFO


# s3://zrive-ds-data/groceries/sampled-datasets/ ->
bucket_name = "zrive-ds-data"
folder = "groceries/sampled-datasets"
load_dotenv("src/module_2/.env")

# AWS S3 Client
s3 = boto3.client(
    service_name="s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
contents = []
try:
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder)
    contents = response.get("Contents", [])
except NoCredentialsError as e:
    logger.info("No credentials AWS found")
    logger.warning(e)
except ClientError as e:
    logger.info("Client error")
    logger.warning(e)
except Exception as e:
    logger.info("Unexpected error")
    logger.warning(e)


# Response return :
"""{'Contents': [
    {'Key': 'groceries/sampled-datasets/orders.parquet', ...},
    {'Key': 'groceries/sampled-datasets/inventory.parquet', ...},
    ...
  ],
  ...
}"""


# Local directory to store data
local_dir = os.path.join(os.path.dirname(__file__), "data")
# __file__ means current file path
# .join -> each comma means a / or \
os.makedirs(local_dir, exist_ok=True)


for content in contents:
    key = content["Key"]
    if not key.endswith("/"):  # it is an archive then
        dataset_name = os.path.basename(key)  # archive name without the path
        destination_path = os.path.join(local_dir, dataset_name)
        try:
            logger.info(f"Downloading {dataset_name}")
            s3.download_file(bucket_name, key, destination_path)
            logger.info(f"Dataset stored in {destination_path}")
        except ClientError as e:
            logger.info("Client error")
            logger.warning(e)
        except Exception as e:
            logger.info("Unexpected error")
            logger.warning(e)
