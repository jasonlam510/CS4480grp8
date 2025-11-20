# Upload Files to HDFS on GCP Dataproc

This guide explains how to upload files from your local machine to HDFS on the GCP Dataproc cluster.

## Prerequisites

- `gcloud` CLI installed and configured
- Access to the GCP project: `cs4480-grp8-478507`
- SSH access to the master node: `cs4480-m`

## Steps

### 1. Zip the Folder Locally

First, create a zip archive of the folder you want to upload:

```bash
zip -r <folder_name>.zip <folder_name>/
```

Replace `<folder_name>` with the actual name of your folder.

### 2. Upload to GCP Master Node

Use `gcloud compute scp` to upload the zip file to the master node:

```bash
gcloud compute scp <folder_name>.zip cs4480-m:~/ \
  --project=cs4480-grp8-478507 \
  --zone=us-central1-a
```

Replace `<folder_name>` with the actual name of your folder.

This will copy the zip file to the home directory (`~/`) of the master node.

### 3. SSH into the Master Node

Connect to the master node:

```bash
gcloud compute ssh cs4480-m \
  --project=cs4480-grp8-478507 \
  --zone=us-central1-a
```

### 4. Unzip on the Master Node

Once connected to the master node, unzip the file:

```bash
unzip <folder_name>.zip
```

Replace `<folder_name>` with the actual name of your folder.

This will extract the contents in the current directory (home directory by default).

### 5. Upload to HDFS

Upload the unzipped folder (or files) to HDFS:

```bash
hdfs dfs -put <folder_name> <hdfs_path>/
```

Replace:
- `<folder_name>` with the actual name of your folder
- `<hdfs_path>` with your desired HDFS path (e.g., `/user/your_username`, `/data`, `/input`)

### 6. Verify Upload

Verify that the files were uploaded successfully:

```bash
hdfs dfs -ls <hdfs_path>/<folder_name>
```

Or check the file size:

```bash
hdfs dfs -du -h <hdfs_path>/<folder_name>
```

Replace `<hdfs_path>` and `<folder_name>` with your actual values.

## Complete Example

Here's a complete example workflow:

```bash
# 1. Zip locally
zip -r <folder_name>.zip <folder_name>/

# 2. Upload to master node
gcloud compute scp <folder_name>.zip cs4480-m:~/ \
  --project=cs4480-grp8-478507 \
  --zone=us-central1-a

# 3. SSH into master node
gcloud compute ssh cs4480-m \
  --project=cs4480-grp8-478507 \
  --zone=us-central1-a

# 4. On master node: Unzip
unzip <folder_name>.zip

# 5. On master node: Upload to HDFS
hdfs dfs -put <folder_name> <hdfs_path>/

# 6. On master node: Verify
hdfs dfs -ls <hdfs_path>/<folder_name>
```

Replace:
- `<folder_name>` with your actual folder name
- `<hdfs_path>` with your desired HDFS path

## Alternative: Direct Upload to HDFS (Small Files)

For small files or single files, you can upload directly to HDFS without zipping:

```bash
# Upload single file directly
gcloud compute scp <file_name> cs4480-m:~/ \
  --project=cs4480-grp8-478507 \
  --zone=us-central1-a

# SSH and put to HDFS
gcloud compute ssh cs4480-m \
  --project=cs4480-grp8-478507 \
  --zone=us-central1-a -- \
  "hdfs dfs -put ~/<file_name> <hdfs_path>/<file_name>"
```

Replace:
- `<file_name>` with your actual file name
- `<hdfs_path>` with your desired HDFS path

## Notes

- The master node is `cs4480-m` in zone `us-central1-a`
- Always verify your uploads to ensure data integrity
- For large files, zipping can reduce transfer time
- Make sure you have write permissions to the HDFS path you're uploading to

