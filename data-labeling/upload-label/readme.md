# Upload Labels to HDFS for PySpark Training

This directory contains scripts to upload image labels (predictions) to HDFS and create a Hive table for PySpark model training.

## Overview

The workflow consists of three main steps:

1. **Transform CSV**: Convert GCS file paths to HDFS paths
2. **Upload to HDFS**: Copy the transformed CSV to HDFS
3. **Create Hive Table**: Create an external Hive table pointing to the CSV

## Prerequisites

- `gcloud` CLI installed and configured
- Access to GCP project
- SSH access to Dataproc master node
- Python 3 with required libraries

## Configuration

**Required:** All scripts require cluster configuration via environment variables or command-line arguments. This ensures scripts work with any Hadoop cluster without hardcoded defaults.

### Environment Variables (Required)

Set these environment variables before running the scripts:

```bash
export DATAPROC_PROJECT="your-project-id"
export DATAPROC_ZONE="us-central1-a"
export DATAPROC_MASTER_NODE="your-master-node-name"
export HDFS_BASE_PATH="/data/frames"  # Optional, defaults to /data/frames
```

**Example:**
```bash
# Set required environment variables
export DATAPROC_PROJECT="cs4480-grp8-478507"
export DATAPROC_ZONE="us-central1-b"
export DATAPROC_MASTER_NODE="test-m"

# Now run scripts - they will use these values
./upload_labels_to_hdfs.sh predictions.csv /data/labels/predictions.csv image_labels
```

**Alternative: Command-Line Arguments**
You can also provide values via command-line arguments (takes precedence over environment variables):

```bash
python3 setup_hive_table.py predictions.csv /data/labels/predictions.csv image_labels \
  --project my-project --zone us-east1-b --master_node my-cluster-m
```

**Note:** If environment variables are not set and command-line arguments are not provided, scripts will fail with an error message indicating what is required.

## Quick Start

Use the orchestration script to run the complete workflow:

```bash
./upload_labels_to_hdfs.sh <input_csv> <hdfs_csv_path> <table_name>
```

**Example:**
```bash
./upload_labels_to_hdfs.sh ../get-label/prediction-model-2025-11-24T09_26_03.018739Z_predictions.csv \
  /data/labels/predictions.csv \
  image_labels
```

## Individual Scripts

### 1. Transform CSV (GCS → HDFS paths)

Transforms the prediction CSV to replace GCS URIs with HDFS paths.

```bash
python3 transform_csv_gcs_to_hdfs.py <input_csv> <output_csv> [--hdfs_base_path]
```

**Example:**
```bash
python3 transform_csv_gcs_to_hdfs.py predictions.csv predictions_hdfs.csv --hdfs_base_path /data/frames
```

**What it does:**
- Reads CSV with GCS paths: `gs://cs4480-thearcticskies/frames/.../image.png`
- Converts to HDFS paths: `hdfs:///data/frames/.../image.png`
- Maintains the `response` column unchanged

### 2. Upload Frames from GCS to HDFS

Copies image frames from GCS to HDFS (optional, if images aren't already in HDFS).

```bash
python3 upload_frames_gcs_to_hdfs.py <gcs_bucket_path> <hdfs_path> [--project] [--zone] [--master_node]
```

**Example:**
```bash
python3 upload_frames_gcs_to_hdfs.py gs://cs4480-thearcticskies/frames /data/frames
```

**Note:** This is a long-running operation that copies all image files. Make sure images are already in HDFS before running the label upload scripts.

### 3. Setup Hive Table

Uploads the transformed CSV to HDFS and creates a Hive external table.

```bash
python3 setup_hive_table.py <csv_file> <hdfs_csv_path> <table_name> [--project] [--zone] [--master_node]
```

**Example:**
```bash
python3 setup_hive_table.py predictions_hdfs.csv /data/labels/predictions.csv image_labels
```

**What it does:**
- Uploads CSV to HDFS
- Creates Hive external table with schema:
  - `file_uri` (STRING): HDFS path to image
  - `response` (STRING): Classification label
- Table uses OpenCSVSerde to read CSV format
- Skips header row automatically

## Hive Table Schema

The created Hive table has the following structure:

```sql
CREATE EXTERNAL TABLE image_labels (
    file_uri STRING,
    response STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
    "separatorChar" = ",",
    "quoteChar" = "\"",
    "escapeChar" = "\\"
)
STORED AS TEXTFILE
LOCATION '/data/labels'
TBLPROPERTIES (
    "skip.header.line.count" = "1"
);
```

## Using in PySpark

Once the Hive table is created, you can use it in PySpark:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Image Classification Training") \
    .enableHiveSupport() \
    .getOrCreate()

# Read from Hive table
df = spark.read.table("image_labels")

# Show sample data
df.show(10)

# Use for training
# df contains columns: file_uri, response
```

## HDFS Paths

Default paths used:
- **Images**: `/data/frames/` (mirrors GCS structure)
- **Labels CSV**: `/data/labels/predictions.csv`

You can customize these paths using the `--hdfs_base_path` argument.

## Cluster Configuration

**Required Configuration** (see [Configuration](#configuration) section above):
- **Project**: Must be set via `$DATAPROC_PROJECT` or `--project` argument
- **Zone**: Must be set via `$DATAPROC_ZONE` or `--zone` argument
- **Master Node**: Must be set via `$DATAPROC_MASTER_NODE` or `--master_node` argument

Configuration priority:
1. Command-line arguments (highest priority)
2. Environment variables
3. Script will fail if neither is provided

## Troubleshooting

### Hive table creation fails

If beeline or hive CLI is not available, you can create the table manually:

1. SSH into the master node:
   ```bash
   gcloud compute ssh cs4480-m --project cs4480-grp8-478507 --zone us-central1-a
   ```

2. Run beeline:
   ```bash
   beeline -u jdbc:hive2://localhost:10000
   ```

3. Execute the CREATE TABLE statement (see Hive Table Schema section above)

### CSV upload fails

Check that:
- You have write permissions to the HDFS path
- The HDFS directory exists (script creates it automatically)
- The CSV file is not corrupted

### Path transformation issues

Verify that:
- Input CSV has `file_uri` column with GCS paths
- GCS paths follow the pattern: `gs://cs4480-thearcticskies/frames/...`
- HDFS base path matches where images are actually stored

## Complete Workflow Example

```bash
# 1. Transform CSV
python3 transform_csv_gcs_to_hdfs.py \
  ../get-label/prediction-model-2025-11-24T09_26_03.018739Z_predictions.csv \
  predictions_hdfs.csv

# 2. Upload CSV and create Hive table
python3 setup_hive_table.py \
  predictions_hdfs.csv \
  /data/labels/predictions.csv \
  image_labels

# 3. Verify in PySpark (on Dataproc)
# spark.read.table("image_labels").show(10)
```

Or use the orchestration script:

```bash
./upload_labels_to_hdfs.sh \
  ../get-label/prediction-model-2025-11-24T09_26_03.018739Z_predictions.csv \
  /data/labels/predictions.csv \
  image_labels
```

