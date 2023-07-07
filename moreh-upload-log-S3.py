#!/usr/bin/env python3

import sys
import os
import boto3
from argparse import ArgumentParser

# awscli credential
ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", None)
SECRET_KEY = os.environ.get("S3_SECRET_KEY", None)


def parse_args():
    parser = ArgumentParser(description="Upload a log file to moreh-vietnam AWS S3")
    parser.add_argument("filename")
    parser.add_argument("--folder", default="")

    args = parser.parse_args()
    return args


def upload(path, directory):
    """Upload a file to moreh-vietnam bucket.

    Args:
        path : file path. It can be a relative path or an absolute path.
    """
    s3 = boto3.resource(
        "s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
    )

    # .txt suffix make us to open a uploaded file to be easily read.
    suffix = ".txt"
    basename = os.path.basename(path)
    if not directory:
        s3_object_key = f"{basename}{suffix}"
    else:
        s3_object_key = f"{directory}/{basename}{suffix}"
    bucket_name = 'moreh-vietnam'

    s3.Bucket(bucket_name).upload_file(
        path,
        s3_object_key,
        ExtraArgs={"ContentType": "text/plain", "ACL": "public-read"},
    )
    s3_url = f"https://{bucket_name}.s3.ap-northeast-2.amazonaws.com/{s3_object_key}"
    print(f"{basename} -> {s3_url}")


if __name__ == "__main__":
    args = parse_args()

    if not ACCESS_KEY or not SECRET_KEY:
        print("Keys for accessing AmazonAWS S3 not found!", file=sys.stderr)
        sys.exit(1)

    upload(args.filename, args.folder)
