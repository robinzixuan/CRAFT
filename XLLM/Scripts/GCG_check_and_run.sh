#!/bin/bash

# Function to check if run_GCG.sh is still running
is_running() {
    pgrep -f run_GCG.sh > /dev/null
    return $?
}

# Loop until run_GCG.sh is no longer running
while is_running; do
    echo "run_GCG.sh is still running. Checking again in 10 minutes..."
    sleep 120  # Check every 2 minutes
done

echo "run_GCG.sh has finished. Starting run_GCG.sh..."
./Scripts/run_GCG.sh
