#!/usr/bin/env bash

export S3_REGION="${AWS_REGION}"
export S3_ACCESS_KEY="${AWS_ACCESS_KEY_ID}"
export S3_SECRET_KEY="${AWS_SECRET_ACCESS_KEY}"
export S3_BUCKET_NAME="${BUCKET_NAME}"
export S3_ENDPOINT_URL="${AWS_ENDPOINT_URL_S3}"
export S3_INPUT_DIR="input"
export S3_OUTPUT_DIR="output"

python fetch_models.py

python waifuwave.py --host=0.0.0.0 --port=8080