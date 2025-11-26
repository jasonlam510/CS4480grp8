#!/bin/bash

# Script to upload a zip path to GCP Dataproc master node, unzip it, upload to HDFS, and clean up
#
# IMPORTANT: Before running this script:
#   1. Create a folder to contain all the files you want to upload
#   2. Add all your files to that folder
#   3. Zip the folder: zip -r <folder_name>.zip <folder_name>/
#
# Usage: ./upload_to_hdfs.sh <zip_path> <hdfs_path>
# Example: ./upload_to_hdfs.sh data.zip /data
# Note: The folder name will be auto-detected from the zip file contents

# Configuration - must be set via environment variables
PROJECT="${DATAPROC_PROJECT}"
ZONE="${DATAPROC_ZONE}"
MASTER_NODE="${DATAPROC_MASTER_NODE}"

# Validate required environment variables
if [ -z "$PROJECT" ]; then
    echo "Error: DATAPROC_PROJECT environment variable is required"
    echo "Set it with: export DATAPROC_PROJECT='your-project-id'"
    exit 1
fi

if [ -z "$ZONE" ]; then
    echo "Error: DATAPROC_ZONE environment variable is required"
    echo "Set it with: export DATAPROC_ZONE='us-central1-a'"
    exit 1
fi

if [ -z "$MASTER_NODE" ]; then
    echo "Error: DATAPROC_MASTER_NODE environment variable is required"
    echo "Set it with: export DATAPROC_MASTER_NODE='your-master-node-name'"
    exit 1
fi

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <zip_path> <hdfs_path>"
    echo "Example: $0 data.zip /data"
    echo ""
    echo "IMPORTANT: Before running this script:"
    echo "  1. Create a folder to contain all the files you want to upload"
    echo "  2. Add all your files to that folder"
    echo "  3. Zip the folder: zip -r <folder_name>.zip <folder_name>/"
    echo ""
    echo "Note: The folder name inside the zip will be auto-detected."
    echo ""
    echo "Required environment variables:"
    echo "  DATAPROC_PROJECT: GCP project ID"
    echo "  DATAPROC_ZONE: GCP zone"
    echo "  DATAPROC_MASTER_NODE: Dataproc master node name"
    exit 1
fi

ZIP_FILE="$1"
HDFS_PATH="$2"
ZIP_BASENAME=$(basename "$ZIP_FILE")
ZIP_MASTER_PATH="~/${ZIP_BASENAME}"

# Validate zip path exists
if [ ! -f "$ZIP_FILE" ]; then
    echo "Error: Zip path '$ZIP_FILE' not found!"
    exit 1
fi

echo "=========================================="
echo "Uploading to HDFS on GCP Dataproc"
echo "=========================================="
echo "Zip path: $ZIP_FILE"
echo "HDFS path: $HDFS_PATH"
echo "Project: $PROJECT"
echo "Zone: $ZONE"
echo "Master node: $MASTER_NODE"
echo ""

# Step 1: Upload zip path to master node
echo "[1/4] Uploading zip path to master node..."
gcloud compute scp "$ZIP_FILE" ${MASTER_NODE}:~/ \
  --project=${PROJECT} \
  --zone=${ZONE}

if [ $? -ne 0 ]; then
    echo "Error: Failed to upload zip path to master node"
    exit 1
fi
echo "✓ Zip path uploaded successfully"
echo ""

# Step 2: SSH into master node, unzip, upload to HDFS, and clean up
echo "[2/4] Unzipping on master node..."
echo "[3/4] Uploading to HDFS..."
echo "[4/4] Cleaning up..."

# Escape the zip basename for safe use in remote command
ZIP_BASENAME_ESCAPED=$(printf '%q' "$ZIP_BASENAME")

gcloud compute ssh ${MASTER_NODE} \
  --project=${PROJECT} \
  --zone=${ZONE} -- \
  "cd ~ && ZIP_BASENAME=${ZIP_BASENAME_ESCAPED} && echo \"Unzipping \$ZIP_BASENAME...\" && unzip -q \"\$ZIP_BASENAME\" && echo '✓ Unzipped successfully' && FOLDER_NAME=\$(ls -d */ 2>/dev/null | head -1 | sed 's|/$||') && if [ -z \"\$FOLDER_NAME\" ] || [ ! -d \"\$FOLDER_NAME\" ]; then echo 'Error: Could not detect folder name from zip'; exit 1; fi && echo \"Detected folder: \$FOLDER_NAME\" && echo 'Uploading to HDFS (this may take a while for large folders)...' && hdfs dfs -mkdir -p ${HDFS_PATH} && hdfs dfs -put \"\$FOLDER_NAME\" ${HDFS_PATH}/ && echo '✓ Uploaded to HDFS successfully' && rm -rf \"\$FOLDER_NAME\" && echo '✓ Cleaned up unzipped folder' && rm -f \"\$ZIP_BASENAME\" && echo '✓ Cleaned up zip path'"

if [ $? -ne 0 ]; then
    echo "Error: Failed to process files on master node"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Upload complete!"
echo "=========================================="
echo "Files are now available in your HDFS home directory at:"
echo "  /user/\$(whoami)${HDFS_PATH}"
echo ""
echo "To verify, run:"
echo "  gcloud compute ssh ${MASTER_NODE} --project=${PROJECT} --zone=${ZONE} -- 'hdfs dfs -ls ${HDFS_PATH}/'"
echo ""
echo "Or check your HDFS home directory:"
echo "  gcloud compute ssh ${MASTER_NODE} --project=${PROJECT} --zone=${ZONE} -- 'hdfs dfs -ls ~/'"

