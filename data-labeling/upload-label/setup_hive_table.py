#!/usr/bin/env python3
"""
Setup Hive Table for Image Labels

Uploads transformed CSV to HDFS and creates a Hive external table for PySpark.

Usage:
    python3 setup_hive_table.py {csv_file} {hdfs_csv_path} {table_name} [--project] [--zone] [--master_node]

Arguments:
    csv_file: Local path to the transformed CSV file
    hdfs_csv_path: HDFS path where CSV will be stored (e.g., /data/labels/predictions.csv)
    table_name: Name for the Hive table (e.g., image_labels)
    --project: GCP project ID (default: cs4480-grp8-478507)
    --zone: GCP zone (default: us-central1-a)
    --master_node: Dataproc master node name (default: cs4480-m)

Example:
    python3 setup_hive_table.py predictions_hdfs.csv /data/labels/predictions.csv image_labels
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(command, check=True):
    """
    Run a shell command and return the result
    
    Args:
        command: Command to run (list of strings)
        check: If True, raise exception on non-zero exit code
        
    Returns:
        CompletedProcess object
    """
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        print(f"stderr: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, command)
    
    return result


def upload_csv_to_hdfs(csv_file: str, hdfs_csv_path: str,
                       project: str,
                       zone: str,
                       master_node: str):
    """
    Upload CSV file to HDFS
    
    Args:
        csv_file: Local path to CSV file
        hdfs_csv_path: HDFS destination path
        project: GCP project ID
        zone: GCP zone
        master_node: Dataproc master node name
    """
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file '{csv_file}' not found")
    
    print(f"Uploading CSV to master node...")
    
    # Step 1: Copy CSV to master node
    scp_cmd = [
        "gcloud", "compute", "scp", str(csv_path),
        f"{master_node}:~/",
        "--project", project,
        "--zone", zone
    ]
    
    run_command(scp_cmd)
    print("✓ CSV copied to master node")
    
    # Step 2: Create HDFS directory and upload
    csv_filename = csv_path.name
    hdfs_dir = str(Path(hdfs_csv_path).parent)
    
    upload_cmd = [
        "gcloud", "compute", "ssh", master_node,
        "--project", project,
        "--zone", zone, "--",
        f"""
        hdfs dfs -mkdir -p {hdfs_dir} && \
        hdfs dfs -put ~/{csv_filename} {hdfs_csv_path} && \
        hdfs dfs -chmod 644 {hdfs_csv_path} && \
        rm -f ~/{csv_filename} && \
        echo "✓ CSV uploaded to HDFS"
        """
    ]
    
    run_command(upload_cmd)
    print(f"✓ CSV uploaded to HDFS: {hdfs_csv_path}")


def create_hive_table(hdfs_csv_path: str, table_name: str,
                      project: str,
                      zone: str,
                      master_node: str):
    """
    Create Hive external table pointing to CSV in HDFS
    
    Args:
        hdfs_csv_path: HDFS path to CSV file
        table_name: Name for the Hive table
        project: GCP project ID
        zone: GCP zone
        master_node: Dataproc master node name
    """
    hdfs_dir = str(Path(hdfs_csv_path).parent)
    
    # Hive DDL to create external table
    # Escape quotes properly for shell command
    hive_ddl_single_line = (
        f"CREATE EXTERNAL TABLE IF NOT EXISTS {table_name} "
        f"(file_uri STRING, response STRING) "
        f"ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' "
        f"WITH SERDEPROPERTIES (\"separatorChar\"=\",\", \"quoteChar\"=\"\\\"\", \"escapeChar\"=\"\\\\\") "
        f"STORED AS TEXTFILE LOCATION '{hdfs_dir}' "
        f"TBLPROPERTIES (\"skip.header.line.count\"=\"1\");"
    )
    
    # Pretty formatted DDL for display
    hive_ddl = f"""CREATE EXTERNAL TABLE IF NOT EXISTS {table_name} (
    file_uri STRING,
    response STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
    "separatorChar" = ",",
    "quoteChar" = "\\"",
    "escapeChar" = "\\\\"
)
STORED AS TEXTFILE
LOCATION '{hdfs_dir}'
TBLPROPERTIES (
    "skip.header.line.count" = "1"
);"""
    
    print(f"Creating Hive table '{table_name}'...")
    
    # Execute Hive DDL using beeline or hive CLI
    # Create a temporary SQL file on master node and execute it
    create_table_cmd = [
        "gcloud", "compute", "ssh", master_node,
        "--project", project,
        "--zone", zone, "--",
        f"""
        cat > /tmp/create_table_{table_name}.sql << 'EOFSQL'
{hive_ddl}
EOFSQL
        beeline -u jdbc:hive2://localhost:10000 -n $(whoami) -f /tmp/create_table_{table_name}.sql 2>&1 || \
        hive -f /tmp/create_table_{table_name}.sql 2>&1
        rm -f /tmp/create_table_{table_name}.sql
        """
    ]
    
    try:
        result = run_command(create_table_cmd, check=False)
        if result.returncode == 0:
            print(f"✓ Hive table '{table_name}' created successfully")
        else:
            print(f"Warning: Command returned non-zero exit code")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            print("\nYou may need to create the table manually.")
            print("SSH into master node and run:")
            print(f"  beeline -u jdbc:hive2://localhost:10000")
            print("\nThen execute:")
            print(hive_ddl)
            # Don't raise exception, just warn
    except Exception as e:
        print(f"Warning: Error creating table: {e}")
        print("\nYou may need to create the table manually using:")
        print(f"  gcloud compute ssh {master_node} --project {project} --zone {zone}")
        print(f"  beeline -u jdbc:hive2://localhost:10000")
        print(f"\nThen run:")
        print(hive_ddl)
        # Don't raise, allow user to create manually
    
    # Verify table
    print(f"\nVerifying table '{table_name}'...")
    verify_cmd = [
        "gcloud", "compute", "ssh", master_node,
        "--project", project,
        "--zone", zone, "--",
        f"""
        beeline -u jdbc:hive2://localhost:10000 -n $(whoami) -e "SELECT COUNT(*) FROM {table_name};" 2>/dev/null || \
        hive -e "SELECT COUNT(*) FROM {table_name};" 2>/dev/null
        """
    ]
    
    try:
        result = run_command(verify_cmd, check=False)
        if result.returncode == 0:
            print("✓ Table verification successful")
            print(f"Query result:\n{result.stdout}")
        else:
            print("Warning: Could not verify table (this is okay if table was just created)")
    except Exception as e:
        print(f"Warning: Verification failed: {e}")


def main():
    """Main function to handle command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Upload CSV to HDFS and create Hive table for PySpark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 setup_hive_table.py predictions_hdfs.csv /data/labels/predictions.csv image_labels
  python3 setup_hive_table.py labels.csv /data/labels/labels.csv frame_predictions --project my-project
        """
    )
    
    parser.add_argument('csv_file',
                       help='Local path to the transformed CSV file')
    parser.add_argument('hdfs_csv_path',
                       help='HDFS path where CSV will be stored (e.g., /data/labels/predictions.csv)')
    parser.add_argument('table_name',
                       help='Name for the Hive table (e.g., image_labels)')
    parser.add_argument('--project', 
                       default=os.getenv('DATAPROC_PROJECT'),
                       required=not bool(os.getenv('DATAPROC_PROJECT')),
                       help='GCP project ID (required: set DATAPROC_PROJECT env var or use --project)')
    parser.add_argument('--zone', 
                       default=os.getenv('DATAPROC_ZONE'),
                       required=not bool(os.getenv('DATAPROC_ZONE')),
                       help='GCP zone (required: set DATAPROC_ZONE env var or use --zone)')
    parser.add_argument('--master_node', 
                       default=os.getenv('DATAPROC_MASTER_NODE'),
                       required=not bool(os.getenv('DATAPROC_MASTER_NODE')),
                       help='Dataproc master node name (required: set DATAPROC_MASTER_NODE env var or use --master_node)')
    
    args = parser.parse_args()
    
    try:
        # Step 1: Upload CSV to HDFS
        print("=" * 60)
        print("Step 1: Uploading CSV to HDFS")
        print("=" * 60)
        upload_csv_to_hdfs(
            csv_file=args.csv_file,
            hdfs_csv_path=args.hdfs_csv_path,
            project=args.project,
            zone=args.zone,
            master_node=args.master_node
        )
        
        print()
        
        # Step 2: Create Hive table
        print("=" * 60)
        print("Step 2: Creating Hive Table")
        print("=" * 60)
        create_hive_table(
            hdfs_csv_path=args.hdfs_csv_path,
            table_name=args.table_name,
            project=args.project,
            zone=args.zone,
            master_node=args.master_node
        )
        
        print()
        print("=" * 60)
        print("✓ Setup complete!")
        print("=" * 60)
        print(f"Hive table '{args.table_name}' is ready for PySpark")
        print(f"\nTo query the table, run:")
        print(f"  gcloud compute ssh {args.master_node} --project {args.project} --zone {args.zone}")
        print(f"  beeline -u jdbc:hive2://localhost:10000")
        print(f"  SELECT * FROM {args.table_name} LIMIT 10;")
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

