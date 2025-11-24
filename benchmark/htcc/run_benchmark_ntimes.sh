#!/bin/bash

# Script to submit benchmark job 6 times sequentially
# Each job waits for the previous one to complete before submitting the next
# 
# Usage:
#   ./run_benchmark_6times.sh                    # Run in foreground
#   nohup ./run_benchmark_6times.sh > run.log 2>&1 &  # Run in background with logging

SUBMIT_FILE="extract_frames.submit"
NUM_RUNS=5

echo "=========================================="
echo "Sequential Benchmark Runner"
echo "Will submit $NUM_RUNS jobs one by one"
echo "Each job will append results to benchmark_results.csv"
echo "=========================================="
echo ""

for i in $(seq 1 $NUM_RUNS); do
    echo "----------------------------------------"
    echo "Run $i of $NUM_RUNS"
    echo "----------------------------------------"
    
    # Submit the job
    echo "Submitting job..."
    JOB_OUTPUT=$(condor_submit $SUBMIT_FILE 2>&1)
    JOB_ID=$(echo "$JOB_OUTPUT" | grep "submitted to cluster" | sed 's/.*cluster \([0-9]*\).*/\1/')
    
    if [ -z "$JOB_ID" ]; then
        echo "ERROR: Failed to submit job. Output:"
        echo "$JOB_OUTPUT"
        exit 1
    fi
    
    echo "Job submitted with ID: $JOB_ID.0"
    echo "Waiting for job to complete..."
    
    # Wait for job to complete
    while true; do
        sleep 10  # Check every 10 seconds
        # Check if job still exists in queue
        condor_q $JOB_ID.0 2>&1 | grep -q "$JOB_ID"
        
        if [ $? -ne 0 ]; then
            echo "Job $JOB_ID.0 completed!"
            break
        fi
        
        # Show current status (quietly)
        condor_q $JOB_ID.0 2>/dev/null | grep "$JOB_ID" | head -1
    done
    
    echo "Run $i completed. Waiting 5 seconds before next submission..."
    echo ""
    sleep 5
done

echo "=========================================="
echo "All $NUM_RUNS runs completed!"
echo "Results are in: benchmark_results.csv"
echo "=========================================="

