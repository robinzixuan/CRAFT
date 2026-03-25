#!/bin/bash

module load cuda/cuda-12.1.0-openmpi-4.1.4
export HF_HOME="/projects/p32013/.cache/"
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

PYTHON_SCRIPT="../Experiments/reasoning_exp.py"
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
EVALUATION="default"
RUN_INDEX=2
ADD_BUDGET=True
BUDGET_NUM="5000"

# GPU
GPU_MEMORY=20000
NUM_GPU_SEARCH=1
NUM_TASKS=128 # Number of tasks to run in parallel

# Dataset paths
HARMFUL_DATASET="../Dataset/harmful.csv"
TARGETS_DATASET="../Dataset/harmful_targets.csv"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --evaluation)
      EVALUATION="$2"
      shift 2
      ;;
    --run_index)
      RUN_INDEX="$2"
      shift 2
      ;;
    --add_budget)
      ADD_BUDGET="$2"
      shift 2
      ;;
    --budget_num)
      BUDGET_NUM="$2"
      shift 2
      ;;
    --gpu_memory)
      GPU_MEMORY="$2"
      shift 2
      ;;
    --num_gpu_search)
      NUM_GPU_SEARCH="$2"
      shift 2
      ;;
    --num_tasks)
      NUM_TASKS="$2"
      shift 2
      ;;
    --harmful_dataset)
      HARMFUL_DATASET="$2"
      shift 2
      ;;
    --targets_dataset)
      TARGETS_DATASET="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [ "ADD_BUDGET" = "True" ]; then
    LOG_PATH="Logs/${MODEL_PATH}/reasoning_budget-${RUN_INDEX}"
else
    LOG_PATH="Logs/${MODEL_PATH}/reasoning-${RUN_INDEX}"
fi



# Create the log directory if it does not exist
mkdir -p "$LOG_PATH"

# Conditional flag for EARLY_STOP
EARLY_STOP_FLAG=""
if [ "$EARLY_STOP" = "True" ]; then
    EARLY_STOP_FLAG="--early_stop"
fi

# Function to find the first available GPU
find_free_gpu() {
    # {0..$NUM_GPU_SEARCH}
    for i in $(seq 0 $NUM_GPU_SEARCH); do
        free_mem=$(nvidia-smi -i $i --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1}')
        if [[ "$free_mem" =~ ^[0-9]+$ ]] && [ "$free_mem" -ge $GPU_MEMORY ]; then
            echo $i
            return
        fi
    done

    echo "-1" # Return -1 if no suitable GPU is found
}
# Start the jobs with GPU assignment
FREE_GPU=-1
# Keep looping until a free GPU is found
while [ $FREE_GPU -eq -1 ]; do
    FREE_GPU=$(find_free_gpu)
    if [ $FREE_GPU -eq -1 ]; then
        sleep 5 # Wait for 5 seconds before trying to find a free GPU again
    fi
done
# Run the Python script on the free GPU
(
      echo "Task $index started on GPU $FREE_GPU."
      echo "CMD: CUDA_VISIBLE_DEVICES=$FREE_GPU python -u $PYTHON_SCRIPT  --target_model $MODEL_PATH $ADD_EOS_FLAG  --evaluation $EVALUATION${BUDGET_NUM:+ --budget_num $BUDGET_NUM} --harmful_dataset $HARMFUL_DATASET --targets_dataset $TARGETS_DATASET  --num_tasks  $NUM_TASKS > ${LOG_PATH}/0.log 2>&1" >> ${LOG_PATH}/0.log
      CUDA_VISIBLE_DEVICES=$FREE_GPU python -u "$PYTHON_SCRIPT"  --target_model $MODEL_PATH $ADD_EOS_FLAG  --evaluation $EVALUATION${BUDGET_NUM:+ --budget_num $BUDGET_NUM} --harmful_dataset "$HARMFUL_DATASET" --targets_dataset "$TARGETS_DATASET"  --num_tasks  $NUM_TASKS > "${LOG_PATH}/0.log" 2>&1
      echo "Task $index on GPU $FREE_GPU finished."
  ) &

# Wait for 30 seconds to give the GPU some time to allocate memory
sleep 30


# Wait for all background jobs to finish
wait
