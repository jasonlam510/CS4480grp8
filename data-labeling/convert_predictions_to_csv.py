#!/usr/bin/env python3
"""
Convert Gemini Batch Predictions JSONL to CSV

Converts a predictions JSONL file from Gemini batch API to a CSV file with
file_uri and response columns.

Usage:
    python3 convert_predictions_to_csv.py {predictions_jsonl_file}

Arguments:
    predictions_jsonl_file: Path to the predictions JSONL file

Example:
    python3 convert_predictions_to_csv.py prediction-model-2025-11-24T09_26_03.018739Z_predictions.jsonl
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def extract_file_uri(request_data: dict) -> str:
    """
    Extract file_uri from the request data
    
    Args:
        request_data: The request dictionary from the JSONL line
        
    Returns:
        File URI string, or empty string if not found
    """
    try:
        contents = request_data.get("contents", [])
        if not contents:
            return ""
        
        # Look through all parts in all contents to find file_data
        for content in contents:
            parts = content.get("parts", [])
            for part in parts:
                file_data = part.get("file_data")
                if file_data and file_data.get("file_uri"):
                    return file_data["file_uri"]
        
        return ""
    except Exception as e:
        print(f"Warning: Error extracting file_uri: {e}", file=sys.stderr)
        return ""


def extract_response(response_data: dict) -> str:
    """
    Extract response text from the response data
    
    Args:
        response_data: The response dictionary from the JSONL line
        
    Returns:
        Response text string, or empty string if not found
    """
    try:
        candidates = response_data.get("candidates", [])
        if not candidates:
            return ""
        
        # Get the first candidate's content
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        
        if not parts:
            return ""
        
        # Get the text from the first part
        text = parts[0].get("text", "")
        # Strip any extra whitespace/newlines
        return text.strip()
    except Exception as e:
        print(f"Warning: Error extracting response: {e}", file=sys.stderr)
        return ""


def convert_jsonl_to_csv(jsonl_file: str, csv_file: str = None) -> str:
    """
    Convert predictions JSONL file to CSV
    
    Args:
        jsonl_file: Path to the input JSONL file
        csv_file: Path to the output CSV file (if None, uses same name with .csv extension)
        
    Returns:
        Path to the created CSV file
        
    Raises:
        FileNotFoundError: If JSONL file doesn't exist
    """
    jsonl_path = Path(jsonl_file)
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Predictions file '{jsonl_file}' not found")
    
    # Generate CSV filename if not provided
    if csv_file is None:
        csv_file = jsonl_path.with_suffix('.csv')
    else:
        csv_file = Path(csv_file)
    
    print(f"Reading predictions from: {jsonl_file}")
    print(f"Writing CSV to: {csv_file}")
    
    rows_processed = 0
    rows_with_errors = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as infile, \
         open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
        
        writer = csv.writer(outfile)
        writer.writerow(['file_uri', 'response'])  # Write header
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            try:
                data = json.loads(line)
                
                # Extract file_uri from request
                request_data = data.get("request", {})
                file_uri = extract_file_uri(request_data)
                
                # Extract response text
                response_data = data.get("response", {})
                response_text = extract_response(response_data)
                
                # Write to CSV
                writer.writerow([file_uri, response_text])
                rows_processed += 1
                
                # Progress indicator for large files
                if rows_processed % 1000 == 0:
                    print(f"  Processed {rows_processed} rows...")
                    
            except json.JSONDecodeError as e:
                rows_with_errors += 1
                print(f"Warning: Error parsing JSON on line {line_num}: {e}", file=sys.stderr)
            except Exception as e:
                rows_with_errors += 1
                print(f"Warning: Error processing line {line_num}: {e}", file=sys.stderr)
    
    print(f"\nConversion complete!")
    print(f"  Total rows processed: {rows_processed}")
    if rows_with_errors > 0:
        print(f"  Rows with errors: {rows_with_errors}")
    print(f"  CSV file: {csv_file}")
    
    return str(csv_file)


def main():
    """Main function to handle command-line arguments and convert JSONL to CSV"""
    parser = argparse.ArgumentParser(
        description="Convert Gemini batch predictions JSONL file to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 convert_predictions_to_csv.py prediction-model-2025-11-24T09_26_03.018739Z_predictions.jsonl
  python3 convert_predictions_to_csv.py data-labeling/predictions.jsonl
        """
    )
    
    parser.add_argument('predictions_jsonl_file', 
                       help='Path to the predictions JSONL file')
    parser.add_argument('--output', '-o', 
                       help='Path to the output CSV file (default: same name with .csv extension)')
    
    args = parser.parse_args()
    
    try:
        output_file = convert_jsonl_to_csv(
            jsonl_file=args.predictions_jsonl_file,
            csv_file=args.output
        )
        
        print(f"\n✓ Success! CSV file created: {output_file}")
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

