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

# Configuration
PROJECT="cs4480-grp8-478507"
ZONE="us-central1-a"
MASTER_NODE="cs4480-m"

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
  "cd ~ && \
   ZIP_BASENAME=${ZIP_BASENAME_ESCAPED} && \
   echo \"Unzipping \$ZIP_BASENAME...\" && \
   unzip -q \"\$ZIP_BASENAME\" && \
   echo '✓ Unzipped successfully' && \
   FOLDER_NAME=\$(ls -d */ 2>/dev/null | head -1 | sed 's|/$||') && \
   if [ -z \"\$FOLDER_NAME\" ] || [ ! -d \"\$FOLDER_NAME\" ]; then \
     echo 'Error: Could not detect folder name from zip' && \
     exit 1 && \
   fi && \
   echo \"Detected folder: \$FOLDER_NAME\" && \
   echo 'Uploading to HDFS (this may take a while for large folders)...' && \
   hdfs dfs -mkdir -p ${HDFS_PATH} && \
   hdfs dfs -put \"\$FOLDER_NAME\" ${HDFS_PATH}/ && \
   echo '✓ Uploaded to HDFS successfully' && \
   rm -rf \"\$FOLDER_NAME\" && \
   echo '✓ Cleaned up unzipped folder' && \
   rm -f \"\$ZIP_BASENAME\" && \
   echo '✓ Cleaned up zip path'"

if [ $? -ne 0 ]; then
    echo "Error: Failed to process files on master node"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Upload complete!"
echo "=========================================="
echo "Files are now available at: ${HDFS_PATH}/<folder_name>"
echo ""
echo "Zip file location on master node (before cleanup): ${ZIP_MASTER_PATH}"
echo ""
echo "To verify, run:"
echo "  gcloud compute ssh ${MASTER_NODE} --project=${PROJECT} --zone=${ZONE} -- 'hdfs dfs -ls ${HDFS_PATH}/'"

