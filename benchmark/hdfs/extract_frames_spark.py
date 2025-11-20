# root@cs4480-1m10w-m:/home/jason/scripts# spark-submit     --name "FFmpeg Frame Extractor (Jason)"     --driver-memory 4G     --executor-memory 5500M     --executor-cores 2     --num-executors 10     frame_extractor_timed.py     --input_path hdfs:///user/jason/raw_arcticskies/TheArcticSkies/     --output_path hdfs:///user/jason/frames     --resize_factor 10     --fps 1

import os
import sys
import subprocess
import argparse
import time
import logging # Import the logging module

# Configure logging at the module level for consistency
# This configuration is used primarily by the driver, but the logging
# system is what Spark intercepts for worker logs as well.
# We will initialize it properly in main()
logger = logging.getLogger(__name__)

# --- SPARK CONFIGURATION (Base only - memory set by spark-submit is preferred) ---
# NOTE: Executor memory is usually best set via the spark-submit command line 
# to avoid parsing issues like the one encountered.
SPARK_CONFIG = {
    "spark.dynamicAllocation.enabled": "false",
    "spark.executor.instances": "10",
    "spark.executor.cores": "2",
    "spark.sql.shuffle.partitions": "54",
}

def format_duration(seconds):
    """Formats seconds into human-readable HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = round(seconds % 60, 2)
    return f"{hours:02}:{minutes:02}:{seconds:05.2f}"

def extract_frames_partition(partition_iterator, resize_factor: float, fps: int, hdfs_output_path: str):
    """
    Function executed on each Spark Executor/Partition.
    It takes a list of HDFS file paths and runs FFmpeg on each one.
    
    CRITICAL CHANGE: If any error occurs during the processing of a single video, 
    the exception is re-raised to immediately fail the Spark task.
    """
    
    # Configure logging for the executor process
    executor_logger = logging.getLogger('executor_task')
    if not executor_logger.handlers:
        executor_logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(levelname)s: (Executor) %(message)s'))
        executor_logger.addHandler(handler)

    
    # Check for FFmpeg and Hadoop client paths
    FFMPEG_BIN = '/opt/conda/default/bin/ffmpeg'
    HADOOP_BIN = 'hadoop'

    # 1. Ensure the worker's local temp directory exists
    local_temp_base = f"/tmp/ffmpeg_frames_{os.getpid()}"
    os.makedirs(local_temp_base, exist_ok=True)
    
    results = []

    for hdfs_video_path in partition_iterator:
        video_filename = os.path.basename(hdfs_video_path)
        video_name_no_ext = os.path.splitext(video_filename)[0]
        
        # Initialize paths for cleanup
        local_staging_path = ""
        local_output_dir = ""

        executor_logger.info(f"Processing video: {video_filename}")
        
        # 2. Define local staging and extraction directories
        local_staging_path = os.path.join(local_temp_base, video_filename) # Path for the video copy
        local_output_dir = os.path.join(local_temp_base, video_name_no_ext) # Path for the extracted frames
        hdfs_destination_dir = os.path.join(hdfs_output_path, video_name_no_ext)
        
        try:
            # Create local directory for extracted frames
            os.makedirs(local_output_dir, exist_ok=True)

            # --- 2.1 STAGE: Download video from HDFS to local worker disk ---
            executor_logger.info(f"Staging video {video_filename} from HDFS to {local_staging_path}...")
            # Use -copyToLocal to get the file
            subprocess.run([HADOOP_BIN, 'fs', '-copyToLocal', hdfs_video_path, local_staging_path], check=True, stderr=subprocess.PIPE)
            executor_logger.info(f"Staging successful. Running FFmpeg...")

            # --- 3. Execute FFmpeg (using the LOCAL path) ---
            ffmpeg_command = [
                FFMPEG_BIN, 
                '-i', local_staging_path, # Use the local path here!
                '-threads', '2', 
                '-r', str(fps), 
            ]
            if resize_factor > 0:
                filter_string = f"scale=iw/{resize_factor}:ih/{resize_factor}"
                ffmpeg_command.extend(['-vf', filter_string])
            ffmpeg_command.extend([
                '-y', 
                os.path.join(local_output_dir, '%06d.png')
            ])
            
            # Run FFmpeg
            subprocess.run(ffmpeg_command, check=True, timeout=3600, stderr=subprocess.PIPE)
            executor_logger.info(f"FFmpeg execution finished.")
            
            # --- 4. VERIFY: Check if frames were created locally ---
            created_frame_names = [f for f in os.listdir(local_output_dir) if f.endswith('.png')]
            
            if not created_frame_names:
                # Raise a specific error if no frames were produced.
                raise ValueError("FFmpeg ran successfully but produced zero output frames. The video file may be corrupt or too short.")

            # Create a list of full local paths for upload
            local_frame_paths = [os.path.join(local_output_dir, name) for name in created_frame_names]


            # --- 5. Clean up the staged video file (to save local disk space) ---
            subprocess.run(['rm', '-f', local_staging_path], check=True)
            executor_logger.info(f"Local video cleaned up. Uploading {len(local_frame_paths)} frames to HDFS...")
            
            # --- 6. Upload extracted frames from local worker disk to HDFS ---
            # IMPORTANT: We explicitly list the files instead of using '*' to avoid the "No such file" error
            subprocess.run([HADOOP_BIN, 'fs', '-mkdir', '-p', hdfs_destination_dir], check=True, stderr=subprocess.PIPE)
            
            # The command is: hadoop fs -put file1 file2 file3 ... destination_dir
            upload_command = [HADOOP_BIN, 'fs', '-put'] + local_frame_paths + [hdfs_destination_dir]
            
            subprocess.run(upload_command, check=True, stderr=subprocess.PIPE)
            
            # 7. Clean up local output directory is handled in finally
            
            executor_logger.info(f"Successfully processed and moved {video_filename}.")
            results.append(f"SUCCESS: {video_filename} -> {hdfs_destination_dir} ({len(local_frame_paths)} frames)")

        except (ValueError, subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
            # --- FATAL ERROR HANDLING: Log and immediately RE-RAISE to fail the Spark task ---
            if isinstance(e, subprocess.CalledProcessError):
                error_message = e.stderr.decode('utf-8', errors='ignore')
                executor_logger.error(f"FATAL ERROR: Subprocess failed for {video_filename}. Terminating task. Output: {error_message}")
                results.append(f"ERROR: {video_filename} failed. Terminating task. Subprocess output: {error_message}")
            elif isinstance(e, ValueError):
                executor_logger.error(f"FATAL ERROR: Processing failed for {video_filename} (Zero frames). Terminating task. Reason: {e}")
                results.append(f"ERROR: {video_filename} failed. Terminating task. Reason: {e}")
            elif isinstance(e, FileNotFoundError):
                executor_logger.critical(f"FATAL ERROR: FFmpeg or Hadoop binary not found. Terminating task.")
                results.append(f"FATAL ERROR: {video_filename} failed. Terminating task. FFmpeg or Hadoop binary not found.")
            else:
                executor_logger.error(f"FATAL UNEXPECTED ERROR processing {video_filename}. Terminating task. Reason: {e}")
                results.append(f"UNEXPECTED ERROR: {video_filename} failed. Terminating task. Reason: {e}")
            
            # CRITICAL STEP: Re-raise the exception to terminate the Spark task immediately.
            raise 
        
        finally:
            # Ensure local cleanup runs even if an error occurred inside the try block
            # Note: check=False is used here to ensure cleanup itself doesn't cause a failure
            if os.path.exists(local_staging_path):
                 subprocess.run(['rm', '-f', local_staging_path], check=False)
            if os.path.exists(local_output_dir):
                 subprocess.run(['rm', '-rf', local_output_dir], check=False)


    # Clean up the base temp directory once the partition is done
    if os.path.exists(local_temp_base):
        subprocess.run(['rm', '-rf', local_temp_base])
        
    yield from results


def main():
    """Parses arguments, initializes Spark, and runs the extraction job."""
    
    # Configure logging for the driver process
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: (Driver) %(message)s')
    
    # --- START TIMING (Only used for early exit timing) ---
    start_time = time.time()
    logger.info(f"Job Start Time (UTC): {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time))}")
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Distributed video frame extraction using PySpark and FFmpeg.")
    parser.add_argument("--input_path", required=True, help="HDFS path containing video files (e.g., hdfs://user/videos)")
    parser.add_argument("--output_path", required=True, help="HDFS path where extracted frames will be saved (e.g., hdfs://user/frames)")
    parser.add_argument("--resize_factor", type=float, default=0.0, help="Resize factor (e.g., 2.0 to halve resolution, 0 for no resize)")
    parser.add_argument("--fps", type=int, default=1, help="Frames Per Second to extract.")
    
    args = parser.parse_args()

    # --- Spark Setup ---
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        logger.error("PySpark is not installed or not in the Python path.")
        sys.exit(1)
        
    # Initialize the builder
    builder = SparkSession.builder \
        .appName("FFmpegFrameExtractor")
    
    # Loop through the dictionary and apply each config using the correct .config() method
    for key, value in SPARK_CONFIG.items():
        builder = builder.config(key, value)
        
    spark = builder.getOrCreate()
    
    sc = spark.sparkContext
    logger.info("Spark Session Initialized")

    # --- 1. Distribute File Paths ---
    try:
        # Use 'hadoop fs -ls -R' for recursive listing and robust path extraction.
        list_command = ['hadoop', 'fs', '-ls', '-R', args.input_path]
        
        ls_output = subprocess.check_output(list_command).decode('utf-8')
        
        video_paths = []
        # Loop through lines to find video files
        for line in ls_output.split('\n'):
            line = line.strip()
            # Check for file line (starts with -) and check for case-insensitive mp4 or mov extension
            if line.startswith('-') and (line.lower().endswith('.mp4') or line.lower().endswith('.mov')):
                # The full path is always the last element in the line split by whitespace
                path_parts = line.split()
                if len(path_parts) > 7: # Ensure it's a valid file line with path
                    video_path = path_parts[-1]
                    video_paths.append(video_path)
        
        if not video_paths:
            logger.critical(f"No video files (case-insensitive .mp4 or .mov) found in {args.input_path}. Check the HDFS path and file extensions.")
            spark.stop()
            # Stop timing and report early exit if no work was done
            end_time = time.time()
            total_duration = end_time - start_time
            print(f"End-to-End Time: {format_duration(total_duration)} (No videos processed)")
            return
            
        logger.info(f"Found {len(video_paths)} videos. Starting distribution...")

    except subprocess.CalledProcessError as e:
        # This catches errors like the HDFS path not existing
        logger.error(f"Error listing HDFS files at {args.input_path}. Command failed: {e.stderr.decode()}")
        spark.stop()
        return
    except Exception as e:
        logger.error(f"AN UNEXPECTED ERROR occurred during HDFS listing: {e}")
        spark.stop()
        return

    paths_rdd = sc.parallelize(video_paths, numSlices=int(SPARK_CONFIG['spark.sql.shuffle.partitions']))

    # --- 2. Execute Distributed Task ---
    logger.info("Starting Frame Extraction (The main work)")
    
    # Collect triggers the execution of the entire Spark job (action)
    results = paths_rdd.mapPartitions(lambda iterator: extract_frames_partition(
        iterator, 
        args.resize_factor, 
        args.fps, 
        args.output_path
    )).collect()

    # --- 3. Final Cleanup and Reporting ---
    print("\n--- Job Processing Complete ---")
    spark.stop()

    # --- FINAL REPORTING (No timing) ---
    print("\n--- Summary ---")
    for res in results:
        print(res)

    success_count = sum(1 for res in results if res.startswith("SUCCESS"))
    error_count = len(results) - success_count
    
    print(f"\nFinal Status: {success_count} videos processed successfully, {error_count} failed.")

if __name__ == "__main__":
    main()