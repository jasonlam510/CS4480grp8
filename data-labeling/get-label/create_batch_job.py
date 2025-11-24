#!/usr/bin/env python3
"""
Batch Job JSON Generator for Gemini API

Creates a batch job JSON file (JSONL format) with requests for image analysis.

Usage:
    python3 create_batch_job.py {prompt_file} {image_urls_file} {output_file} {temperature}

Arguments:
    prompt_file: Path to the text file containing the prompt
    image_urls_file: Path to the text file containing image URLs (one per line, gs:// format)
    output_file: Path to the output JSONL file
    temperature: Temperature for generation config (e.g., 0.4)

Example:
    python3 create_batch_job.py prompt.txt frames_urls.txt batch_job.jsonl 0.4
""" 

import argparse
import json
import sys
from pathlib import Path


def get_mime_type(file_uri: str) -> str:
    """
    Determine MIME type from file extension
    
    Args:
        file_uri: GCS file URI (gs://bucket/path/file.ext)
        
    Returns:
        MIME type string (defaults to 'image/jpeg')
    """
    file_uri_lower = file_uri.lower()
    if file_uri_lower.endswith('.png'):
        return 'image/png'
    elif file_uri_lower.endswith('.jpeg') or file_uri_lower.endswith('.jpg'):
        return 'image/jpeg'
    elif file_uri_lower.endswith('.gif'):
        return 'image/gif'
    elif file_uri_lower.endswith('.webp'):
        return 'image/webp'
    elif file_uri_lower.endswith('.bmp'):
        return 'image/bmp'
    else:
        # Default to jpeg if unknown
        return 'image/jpeg'


def read_prompt(prompt_file: str) -> str:
    """
    Read the prompt from the specified text file
    
    Args:
        prompt_file: Path to the prompt text file
        
    Returns:
        Prompt text as string
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
            if not prompt:
                raise ValueError(f"Prompt file '{prompt_file}' is empty")
            return prompt
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file '{prompt_file}' not found")
    except Exception as e:
        raise Exception(f"Error reading prompt file: {e}")


def read_image_urls(image_urls_file: str) -> list:
    """
    Read image URLs from the specified text file (one per line)
    
    Args:
        image_urls_file: Path to the file containing image URLs
        
    Returns:
        List of image URLs (stripped of whitespace, empty lines filtered)
        
    Raises:
        FileNotFoundError: If URLs file doesn't exist
    """
    try:
        urls = []
        with open(image_urls_file, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url:  # Skip empty lines
                    urls.append(url)
        
        if not urls:
            raise ValueError(f"No URLs found in '{image_urls_file}'")
        
        return urls
    except FileNotFoundError:
        raise FileNotFoundError(f"Image URLs file '{image_urls_file}' not found")
    except Exception as e:
        raise Exception(f"Error reading image URLs file: {e}")


def create_batch_job(prompt: str, image_urls: list, 
                     temperature: float = 0.4, output_file: str = None) -> str:
    """
    Create a batch job JSON file (JSONL format)
    
    Args:
        prompt: The prompt text to use for all requests
        image_urls: List of image URLs (gs:// format)
        temperature: Temperature for generation config (default: 0.4)
        output_file: Path to output file (if None, uses default name)
        
    Returns:
        Path to the created output file
    """
    if output_file is None:
        output_file = "batch_job.jsonl"
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating batch job file: {output_file}")
    print(f"Number of requests: {len(image_urls)}")
    print(f"Temperature: {temperature}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, image_uri in enumerate(image_urls, 1):
            # Determine MIME type from file extension
            mime_type = get_mime_type(image_uri)
            
            # Create request JSON object
            request_obj = {
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": prompt},
                                {
                                    "file_data": {
                                        "file_uri": image_uri,
                                        "mime_type": mime_type
                                    }
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": temperature
                    }
                }
            }
            
            # Write as JSON line (JSONL format)
            f.write(json.dumps(request_obj) + '\n')
            
            # Progress indicator for large files
            if i % 1000 == 0:
                print(f"  Processed {i}/{len(image_urls)} requests...")
    
    print(f"\nBatch job file created successfully: {output_file}")
    print(f"Total requests: {len(image_urls)}")
    
    return output_file


def main():
    """Main function to handle command-line arguments and create batch job"""
    parser = argparse.ArgumentParser(
        description="Create a batch job JSON file for Gemini API image analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 create_batch_job.py prompt.txt frames_urls.txt batch_job.jsonl 0.4
  python3 create_batch_job.py prompt.txt frames_urls.txt output.jsonl 0.6
        """
    )
    
    parser.add_argument('prompt_file', help='Path to the text file containing the prompt')
    parser.add_argument('image_urls_file', help='Path to the text file containing image URLs (one per line)')
    parser.add_argument('output_file', help='Path to the output JSONL file')
    parser.add_argument('temperature', type=float,
                       help='Temperature for generation config (e.g., 0.4)')
    
    args = parser.parse_args()
    
    try:
        # Read prompt
        print(f"Reading prompt from: {args.prompt_file}")
        prompt = read_prompt(args.prompt_file)
        print(f"Prompt length: {len(prompt)} characters\n")
        
        # Read image URLs
        print(f"Reading image URLs from: {args.image_urls_file}")
        image_urls = read_image_urls(args.image_urls_file)
        print(f"Found {len(image_urls)} image URLs\n")
        
        # Create batch job
        output_file = create_batch_job(
            prompt=prompt,
            image_urls=image_urls,
            temperature=args.temperature,
            output_file=args.output_file
        )
        
        print(f"\n✓ Success! Batch job file created: {output_file}")
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()