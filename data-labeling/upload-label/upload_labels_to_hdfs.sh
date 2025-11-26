#!/bin/bash

# Script to upload image labels to HDFS and create Hive table
#
# This script orchestrates the complete workflow:
# 1. Transform CSV from GCS paths to HDFS paths
# 2. Upload CSV to HDFS
# 3. Create Hive table
#
# Usage: ./upload_labels_to_hdfs.sh <input_csv> <hdfs_csv_path> <table_name> [hdfs_base_path]
# Example: ./upload_labels_to_hdfs.sh predictions.csv /data/labels/predictions.csv image_labels

# Configuration - must be set via environment variables
PROJECT="${DATAPROC_PROJECT}"
ZONE="${DATAPROC_ZONE}"
MASTER_NODE="${DATAPROC_MASTER_NODE}"
HDFS_BASE_PATH="${HDFS_BASE_PATH:-/data/frames}"

# Validate required environment variables
if [ -z "$PROJECT" ]; then
    echo -e "${RED}Error: DATAPROC_PROJECT environment variable is required${NC}"
    echo "Set it with: export DATAPROC_PROJECT='your-project-id'"
    exit 1
fi

if [ -z "$ZONE" ]; then
    echo -e "${RED}Error: DATAPROC_ZONE environment variable is required${NC}"
    echo "Set it with: export DATAPROC_ZONE='us-central1-a'"
    exit 1
fi

if [ -z "$MASTER_NODE" ]; then
    echo -e "${RED}Error: DATAPROC_MASTER_NODE environment variable is required${NC}"
    echo "Set it with: export DATAPROC_MASTER_NODE='your-master-node-name'"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if correct number of arguments provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <input_csv> <hdfs_csv_path> <table_name> [hdfs_base_path]"
    echo "Example: $0 predictions.csv /data/labels/predictions.csv image_labels"
    echo ""
    echo "Arguments:"
    echo "  input_csv:      Path to CSV file with GCS paths"
    echo "  hdfs_csv_path:  HDFS path where CSV will be stored"
    echo "  table_name:     Name for the Hive table"
    echo "  hdfs_base_path: Base HDFS path for images (default: /data/frames)"
    exit 1
fi

INPUT_CSV="$1"
HDFS_CSV_PATH="$2"
TABLE_NAME="$3"
HDFS_BASE_PATH="${4:-$HDFS_BASE_PATH}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFORM_SCRIPT="${SCRIPT_DIR}/transform_csv_gcs_to_hdfs.py"
SETUP_SCRIPT="${SCRIPT_DIR}/setup_hive_table.py"

# Validate input CSV exists
if [ ! -f "$INPUT_CSV" ]; then
    echo -e "${RED}Error: Input CSV file '$INPUT_CSV' not found!${NC}"
    exit 1
fi

# Generate transformed CSV filename
INPUT_BASENAME=$(basename "$INPUT_CSV")
INPUT_DIR=$(dirname "$INPUT_CSV")
TRANSFORMED_CSV="${INPUT_DIR}/${INPUT_BASENAME%.*}_hdfs.${INPUT_BASENAME##*.}"

echo "=========================================="
echo "Upload Labels to HDFS and Create Hive Table"
echo "=========================================="
echo "Input CSV: $INPUT_CSV"
echo "Transformed CSV: $TRANSFORMED_CSV"
echo "HDFS CSV path: $HDFS_CSV_PATH"
echo "Hive table name: $TABLE_NAME"
echo "HDFS base path: $HDFS_BASE_PATH"
echo ""

# Step 1: Transform CSV
echo -e "${YELLOW}[1/3] Transforming CSV (GCS paths -> HDFS paths)...${NC}"
if [ ! -f "$TRANSFORM_SCRIPT" ]; then
    echo -e "${RED}Error: Transform script not found at '$TRANSFORM_SCRIPT'${NC}"
    exit 1
fi

python3 "$TRANSFORM_SCRIPT" "$INPUT_CSV" "$TRANSFORMED_CSV" --hdfs_base_path "$HDFS_BASE_PATH"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: CSV transformation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ CSV transformed successfully${NC}"
echo ""

# Step 2: Upload CSV to HDFS and create Hive table
echo -e "${YELLOW}[2/3] Uploading CSV to HDFS and creating Hive table...${NC}"
if [ ! -f "$SETUP_SCRIPT" ]; then
    echo -e "${RED}Error: Setup script not found at '$SETUP_SCRIPT'${NC}"
    exit 1
fi

python3 "$SETUP_SCRIPT" "$TRANSFORMED_CSV" "$HDFS_CSV_PATH" "$TABLE_NAME" \
    --project "$PROJECT" \
    --zone "$ZONE" \
    --master_node "$MASTER_NODE"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: HDFS upload or Hive table creation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ CSV uploaded and Hive table created${NC}"
echo ""

# Step 3: Cleanup (optional - keep transformed CSV for reference)
echo -e "${YELLOW}[3/3] Cleanup...${NC}"
read -p "Delete transformed CSV file? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f "$TRANSFORMED_CSV"
    echo -e "${GREEN}✓ Transformed CSV deleted${NC}"
else
    echo "Transformed CSV kept at: $TRANSFORMED_CSV"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Upload complete!${NC}"
echo "=========================================="
echo "Hive table '$TABLE_NAME' is ready for PySpark"
echo ""
echo "To query the table, run:"
echo "  gcloud compute ssh $MASTER_NODE --project $PROJECT --zone $ZONE"
echo "  beeline -u jdbc:hive2://localhost:10000"
echo "  SELECT * FROM $TABLE_NAME LIMIT 10;"
echo ""
echo "To use in PySpark:"
echo "  spark.read.table('$TABLE_NAME').show()"

