#!/bin/bash

# Script to extract frames from all MP4 videos in an input folder
# Usage: ./extract_frames.sh -i <input_folder> -o <output_folder> [options]
# Example: ./extract_frames.sh -i /videos -o /frames -r 2.0 -f 1 -t 2 -p 4

# Don't use set -e as we want to continue processing even if one video fails

# Default values
RESIZE_FACTOR=0
FPS=1
THREADS=0
PARALLEL=1
FFMPEG_BIN="ffmpeg"
BENCHMARK_MODE=false
BENCHMARK_CSV="benchmark_results.csv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print usage
usage() {
    echo "Usage: $0 -i <input_folder> -o <output_folder> [options]"
    echo ""
    echo "Required arguments:"
    echo "  -i, --input_path    Input folder containing MP4 videos"
    echo "  -o, --output_path   Output folder for extracted frames"
    echo ""
    echo "Optional arguments:"
    echo "  -r, --resize_factor Resize factor (e.g., 2.0 to halve resolution, 0 for no resize) [default: 0]"
    echo "  -f, --fps           Frames per second to extract [default: 1]"
    echo "  -t, --threads       Number of threads for FFmpeg [default: 0]"
    echo "  -p, --parallel      Number of videos to process in parallel [default: 1]"
    echo "  --ffmpeg_path       Custom FFmpeg path [default: system ffmpeg]"
    echo "  --benchmark         Run benchmark mode: test multiple parallel job counts [default: false]"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -i /videos -o /frames -r 2.0 -f 1 -t 2 -p 4"
    exit 1
}

# Parse arguments
INPUT_PATH=""
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input_path)
            INPUT_PATH="$2"
            shift 2
            ;;
        -o|--output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -r|--resize_factor)
            RESIZE_FACTOR="$2"
            shift 2
            ;;
        -f|--fps)
            FPS="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        -p|--parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --ffmpeg_path)
            FFMPEG_BIN="$2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK_MODE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo -e "${RED}Error: Input path and output path are required${NC}"
    usage
fi

# Validate input path
if [ ! -d "$INPUT_PATH" ]; then
    echo -e "${RED}Error: Input path '$INPUT_PATH' does not exist or is not a directory${NC}"
    exit 1
fi

# Create output path if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Validate FFmpeg
if ! command -v "$FFMPEG_BIN" &> /dev/null; then
    echo -e "${RED}Error: FFmpeg not found at '$FFMPEG_BIN'${NC}"
    echo "Please install FFmpeg or specify a custom path with --ffmpeg_path"
    exit 1
fi

# Validate numeric arguments
if ! [[ "$RESIZE_FACTOR" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo -e "${RED}Error: Resize factor must be a number >= 0${NC}"
    exit 1
fi

# Check if resize_factor is negative (simple string comparison for 0)
if [[ "$RESIZE_FACTOR" =~ ^- ]]; then
    echo -e "${RED}Error: Resize factor must be >= 0${NC}"
    exit 1
fi

if ! [[ "$FPS" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo -e "${RED}Error: FPS must be a number > 0${NC}"
    exit 1
fi

# Check if FPS is zero or negative
if (( $(echo "$FPS <= 0" | bc -l 2>/dev/null || echo "1") )); then
    # Fallback check if bc is not available
    if ! command -v bc &> /dev/null; then
        # Simple integer check
        if [[ "$FPS" =~ ^0+\.?0*$ ]] || [[ "$FPS" =~ ^- ]]; then
            echo -e "${RED}Error: FPS must be > 0${NC}"
            exit 1
        fi
    else
        if (( $(echo "$FPS <= 0" | bc -l) )); then
            echo -e "${RED}Error: FPS must be > 0${NC}"
            exit 1
        fi
    fi
fi

if ! [[ "$THREADS" =~ ^[0-9]+$ ]] || [ "$THREADS" -lt 0 ]; then
    echo -e "${RED}Error: Threads must be a non-negative integer (0 = use all available cores)${NC}"
    exit 1
fi

if ! [[ "$PARALLEL" =~ ^[0-9]+$ ]] || [ "$PARALLEL" -le 0 ]; then
    echo -e "${RED}Error: Parallel must be a positive integer${NC}"
    exit 1
fi

# Find all MP4 files recursively (bash 3.2 compatible)
echo "Searching for MP4 files in: $INPUT_PATH"
VIDEO_FILES=()
# Use a temporary file to avoid process substitution issues in bash 3.2
TEMP_FILE=$(mktemp)
find "$INPUT_PATH" -type f -iname "*.mp4" | sort > "$TEMP_FILE"

while IFS= read -r file; do
    [ -n "$file" ] && VIDEO_FILES+=("$file")
done < "$TEMP_FILE"
rm -f "$TEMP_FILE"

if [ ${#VIDEO_FILES[@]} -eq 0 ]; then
    echo -e "${YELLOW}Warning: No MP4 files found in '$INPUT_PATH'${NC}"
    exit 0
fi

TOTAL_VIDEOS=${#VIDEO_FILES[@]}
echo -e "${GREEN}Found $TOTAL_VIDEOS MP4 video(s)${NC}"
echo ""

# Function to write CSV header if file doesn't exist
write_csv_header() {
    local csv_file="$1"
    if [ ! -f "$csv_file" ]; then
        echo "parallel_jobs,total_time_seconds,num_videos,total_frames,success_count,fail_count,timestamp" > "$csv_file"
    fi
}

# Function to append results to CSV
append_csv_result() {
    local csv_file="$1"
    local parallel_jobs="$2"
    local total_time="$3"
    local num_videos="$4"
    local total_frames="$5"
    local success_count="$6"
    local fail_count="$7"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "$parallel_jobs,$total_time,$num_videos,$total_frames,$success_count,$fail_count,$timestamp" >> "$csv_file"
}

# Function to count total frames in output directory
count_total_frames() {
    local output_path="$1"
    find "$output_path" -name "*.png" 2>/dev/null | wc -l | tr -d ' '
}

# Function to process a single video
process_video() {
    local video_path="$1"
    local quiet_mode="${2:-false}"
    local video_name=$(basename "$video_path" .mp4)
    local video_name_no_ext=$(basename "$video_path" | sed 's/\.[^.]*$//')
    local output_dir="$OUTPUT_PATH/$video_name_no_ext"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Build FFmpeg command
    local ffmpeg_cmd=(
        "$FFMPEG_BIN"
        "-i" "$video_path"
        "-threads" "$THREADS"
        "-r" "$FPS"
    )
    
    # Add resize filter if resize_factor > 0
    # Use awk for floating point comparison (more portable than bc)
    if command -v awk &> /dev/null; then
        if awk "BEGIN {exit !($RESIZE_FACTOR > 0)}"; then
            ffmpeg_cmd+=("-vf" "scale=iw/$RESIZE_FACTOR:ih/$RESIZE_FACTOR")
        fi
    else
        # Fallback: simple string comparison
        if [[ "$RESIZE_FACTOR" != "0" ]] && [[ "$RESIZE_FACTOR" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            ffmpeg_cmd+=("-vf" "scale=iw/$RESIZE_FACTOR:ih/$RESIZE_FACTOR")
        fi
    fi
    
    # Add output
    ffmpeg_cmd+=("-y" "$output_dir/%03d.png")
    
    # Execute FFmpeg
    if "${ffmpeg_cmd[@]}" > /dev/null 2>&1; then
        # Count extracted frames
        local frame_count=$(find "$output_dir" -name "*.png" | wc -l | tr -d ' ')
        if [ "$quiet_mode" = "false" ]; then
            echo -e "${GREEN}✓${NC} $video_name -> $frame_count frames"
        fi
        return 0
    else
        if [ "$quiet_mode" = "false" ]; then
            echo -e "${RED}✗${NC} $video_name -> Failed"
        fi
        return 1
    fi
}

# Main processing function that returns metrics
run_extraction() {
    local parallel_jobs="$1"
    local quiet_mode="${2:-false}"  # Optional: suppress output for benchmark mode
    
    # Reset counters
    local success_count=0
    local fail_count=0
    local current_jobs=0
    local video_index=0
    declare -a pids=()
    
    # Start timing
    local start_time=$(date +%s.%N)
    
    if [ "$quiet_mode" = "false" ]; then
        echo "Processing videos..."
        echo "Configuration: FPS=$FPS, Threads=$THREADS, Resize=$RESIZE_FACTOR, Parallel=$parallel_jobs"
        echo ""
    fi
    
    # Function to wait for jobs and update counters
    wait_for_jobs() {
        while [ $current_jobs -ge $parallel_jobs ]; do
            if [ ${#pids[@]} -gt 0 ]; then
                wait "${pids[0]}"
                local exit_code=$?
                pids=("${pids[@]:1}")
            else
                wait
                local exit_code=$?
            fi
            current_jobs=$((current_jobs - 1))
            
            if [ $exit_code -eq 0 ]; then
                success_count=$((success_count + 1))
            else
                fail_count=$((fail_count + 1))
            fi
        done
    }
    
    # Process all videos
    for video_file in "${VIDEO_FILES[@]}"; do
        video_index=$((video_index + 1))
        video_name=$(basename "$video_file")
        
        # Wait if we've reached max parallel jobs
        wait_for_jobs
        
        # Show progress (only if not in quiet mode)
        if [ "$quiet_mode" = "false" ]; then
            if [ $parallel_jobs -gt 1 ]; then
                echo "[$video_index/$TOTAL_VIDEOS] Processing: $video_name [Jobs: $current_jobs/$parallel_jobs]"
            else
                echo "[$video_index/$TOTAL_VIDEOS] Processing: $video_name"
            fi
        fi
        
        # Start processing in background if parallel > 1
        if [ $parallel_jobs -gt 1 ]; then
            (process_video "$video_file" "$quiet_mode") &
            local pid=$!
            pids+=($pid)
            current_jobs=$((current_jobs + 1))
        else
            # Sequential processing
            if process_video "$video_file" "$quiet_mode"; then
                success_count=$((success_count + 1))
            else
                fail_count=$((fail_count + 1))
            fi
        fi
    done
    
    # Wait for all remaining background jobs
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
            local exit_code=$?
            current_jobs=$((current_jobs - 1))
            
            if [ $exit_code -eq 0 ]; then
                success_count=$((success_count + 1))
            else
                fail_count=$((fail_count + 1))
            fi
        fi
    done
    
    # Wait for any remaining jobs (fallback)
    while [ $current_jobs -gt 0 ]; do
        wait
        current_jobs=$((current_jobs - 1))
    done
    
    # End timing
    local end_time=$(date +%s.%N)
    local total_time=$(echo "$end_time - $start_time" | bc)
    
    # Count total frames
    local total_frames=$(count_total_frames "$OUTPUT_PATH")
    
    # Return metrics (we'll use global variables to return multiple values)
    BENCHMARK_SUCCESS=$success_count
    BENCHMARK_FAIL=$fail_count
    BENCHMARK_TIME=$total_time
    BENCHMARK_FRAMES=$total_frames
}

# Check if benchmark mode is enabled
if [ "$BENCHMARK_MODE" = "true" ]; then
    echo "=========================================="
    echo "BENCHMARK MODE"
    echo "=========================================="
    echo "Testing parallel job counts: 1, 2, 4, 6, 8, 10"
    echo "Results will be saved to: $BENCHMARK_CSV"
    echo ""
    
    # Initialize CSV file with header
    write_csv_header "$BENCHMARK_CSV"
    
    # Test different parallel job counts
    for parallel_jobs in 1 2 3 4 5 6 7 8 9 10; do
        echo "----------------------------------------"
        echo "Testing with $parallel_jobs parallel job(s)..."
        echo "----------------------------------------"
        
        # Clear output directory before each run (optional - comment out if you want to keep frames)
        # rm -rf "$OUTPUT_PATH"/*
        # mkdir -p "$OUTPUT_PATH"
        
        # Run extraction in quiet mode
        run_extraction "$parallel_jobs" "true"
        
        # Get results from global variables
        success_count=$BENCHMARK_SUCCESS
        fail_count=$BENCHMARK_FAIL
        total_time=$BENCHMARK_TIME
        total_frames=$BENCHMARK_FRAMES
        
        # Append to CSV
        append_csv_result "$BENCHMARK_CSV" "$parallel_jobs" "$total_time" "$TOTAL_VIDEOS" "$total_frames" "$success_count" "$fail_count"
        
        echo "Completed: $total_time seconds"
        echo "Success: $success_count, Failed: $fail_count, Frames: $total_frames"
        echo ""
    done
    
    echo "=========================================="
    echo "Benchmark complete!"
    echo "Results saved to: $BENCHMARK_CSV"
    echo "=========================================="
else
    # Normal mode - run extraction once
    run_extraction "$PARALLEL" "false"
    
    # Print summary
    echo ""
    echo "=========================================="
    echo "Summary"
    echo "=========================================="
    echo "Total videos found: $TOTAL_VIDEOS"
    echo -e "${GREEN}Successfully processed: $BENCHMARK_SUCCESS${NC}"
    if [ $BENCHMARK_FAIL -gt 0 ]; then
        echo -e "${RED}Failed: $BENCHMARK_FAIL${NC}"
    fi
    echo "Total frames extracted: $BENCHMARK_FRAMES"
    echo "Total time: ${BENCHMARK_TIME} seconds"
    echo ""
    echo "Frames extracted to: $OUTPUT_PATH"
fi

