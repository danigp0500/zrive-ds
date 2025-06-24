import os
import logging
from typing import List
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv


# ----------------------
# Logging configuration
# ----------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# ----------------------
# Load AWS credentials
# ----------------------
load_dotenv("src/module_2/.env")

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


def get_s3_client():
    return boto3.client(
        service_name="s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )


# ----------------------
# S3 download utilities
# ----------------------


def list_s3_files(bucket: str, prefix: str) -> List[str]:
    """List all file keys in an S3 prefix (excluding folders)."""
    s3 = get_s3_client()
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = response.get("Contents", [])
        return [obj["Key"] for obj in contents if not obj["Key"].endswith("/")]
    except (NoCredentialsError, ClientError) as e:
        logger.warning(f"S3 listing error: {e}")
        return []
    except Exception as e:
        logger.warning(f"Unexpected error while listing: {e}")
        return []


def download_s3_files(bucket: str, prefix: str, local_subdir: str = "data") -> None:
    """
    Download all S3 files from `prefix` into local folder `../data/local_subdir/`.
    """
    # Path one level above current file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    local_dir = os.path.join(base_dir, local_subdir)
    os.makedirs(local_dir, exist_ok=True)

    s3 = get_s3_client()
    keys = list_s3_files(bucket, prefix)

    for key in keys:
        filename = os.path.basename(key)
        local_path = os.path.join(local_dir, filename)

        if os.path.exists(local_path):
            logger.info(f"{filename} already exists locally. Skipping download.")
            continue

        try:
            logger.info(f"Downloading {key} to {local_path}")
            s3.download_file(bucket, key, local_path)
            logger.info(f"Stored: {local_path}")
        except Exception as e:
            logger.warning(f"Error downloading {key}: {e}")
