
import vllm.distributed.parallel_state as vllm_ps
print(dir(vllm_ps))
try:
    print(f"TP Group: {vllm_ps.get_tensor_model_parallel_group()}")
except Exception as e:
    print(f"Error getting TP group: {e}")
