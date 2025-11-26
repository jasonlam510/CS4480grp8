#!/usr/bin/env python3
"""
Transform CSV: Convert GCS paths to HDFS paths

Transforms the prediction CSV file to replace GCS file URIs with HDFS paths
for use in PySpark training.

Usage:
    python3 transform_csv_gcs_to_hdfs.py {input_csv} {output_csv} [--hdfs_base_path]

Arguments:
    input_csv: Path to the input CSV file with GCS paths
    output_csv: Path to the output CSV file with HDFS paths
    --hdfs_base_path: Base HDFS path for images (default: /data/frames)

Example:
    python3 transform_csv_gcs_to_hdfs.py predictions.csv predictions_hdfs.csv
"""

import argparse
import csv
import sys
from pathlib import Path


def gcs_to_hdfs_path(gcs_uri: str, hdfs_base_path: str = "/data/frames") -> str:
    """
    Convert GCS URI to HDFS path
    
    Args:
        gcs_uri: GCS URI (e.g., "gs://cs4480-thearcticskies/frames/path/to/image.png")
        hdfs_base_path: Base HDFS path (default: /data/frames)
        
    Returns:
        HDFS path (e.g., "hdfs:///data/frames/path/to/image.png")
    """
    # Remove quotes if present
    gcs_uri = gcs_uri.strip('"')
    
    # Extract the path after the bucket/frames prefix
    # Pattern: gs://cs4480-thearcticskies/frames/...
    if "gs://" in gcs_uri:
        # Find the /frames/ part and get everything after it
        if "/frames/" in gcs_uri:
            relative_path = gcs_uri.split("/frames/", 1)[1]
        else:
            # Fallback: try to extract path after bucket name
            parts = gcs_uri.replace("gs://", "").split("/", 2)
            if len(parts) >= 3:
                relative_path = parts[2]
            else:
                raise ValueError(f"Cannot parse GCS URI: {gcs_uri}")
    else:
        # Assume it's already a relative path
        relative_path = gcs_uri
    
    # Construct HDFS path
    hdfs_path = f"hdfs://{hdfs_base_path.rstrip('/')}/{relative_path}"
    return hdfs_path


def transform_csv(input_csv: str, output_csv: str, hdfs_base_path: str = "/data/frames"):
    """
    Transform CSV file from GCS paths to HDFS paths
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        hdfs_base_path: Base HDFS path for images
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV file '{input_csv}' not found")
    
    print(f"Reading CSV from: {input_csv}")
    print(f"Writing transformed CSV to: {output_csv}")
    print(f"HDFS base path: {hdfs_base_path}")
    
    rows_processed = 0
    rows_with_errors = 0
    
    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=['file_uri', 'response'])
        writer.writeheader()
        
        for row in reader:
            try:
                gcs_uri = row['file_uri']
                response = row['response']
                
                # Transform GCS path to HDFS path
                hdfs_path = gcs_to_hdfs_path(gcs_uri, hdfs_base_path)
                
                # Write transformed row
                writer.writerow({
                    'file_uri': hdfs_path,
                    'response': response
                })
                
                rows_processed += 1
                
                # Progress indicator
                if rows_processed % 1000 == 0:
                    print(f"  Processed {rows_processed} rows...")
                    
            except Exception as e:
                rows_with_errors += 1
                print(f"Warning: Error processing row {rows_processed + 1}: {e}", file=sys.stderr)
    
    print(f"\nTransformation complete!")
    print(f"  Total rows processed: {rows_processed}")
    if rows_with_errors > 0:
        print(f"  Rows with errors: {rows_with_errors}")
    print(f"  Output CSV: {output_csv}")


def main():
    """Main function to handle command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Transform CSV file from GCS paths to HDFS paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 transform_csv_gcs_to_hdfs.py predictions.csv predictions_hdfs.csv
  python3 transform_csv_gcs_to_hdfs.py predictions.csv predictions_hdfs.csv --hdfs_base_path /data/frames
        """
    )
    
    parser.add_argument('input_csv', help='Path to the input CSV file with GCS paths')
    parser.add_argument('output_csv', help='Path to the output CSV file with HDFS paths')
    parser.add_argument('--hdfs_base_path', default='/data/frames',
                       help='Base HDFS path for images (default: /data/frames)')
    
    args = parser.parse_args()
    
    try:
        transform_csv(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            hdfs_base_path=args.hdfs_base_path
        )
        
        print(f"\n✓ Success! Transformed CSV created: {args.output_csv}")
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

