#!/bin/bash

# Function to check if run_GPTFuzzer.sh is still running
is_running() {
    pgrep -f run_GPTFuzzer.sh > /dev/null
    return $?
}

# Loop until run_GPTFuzzer.sh is no longer running
while is_running; do
    echo "run_GPTFuzzer.sh is still running. Checking again in 10 minutes..."
    sleep 120  # Check every 2 minutes
done

echo "run_GPTFuzzer.sh has finished. Starting run_GPTFuzzer.sh..."
./Scripts/run_GPTFuzzer.sh
