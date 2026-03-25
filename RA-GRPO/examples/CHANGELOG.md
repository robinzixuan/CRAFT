# Changelog

## [2025-12-28] - 环境适配与训练脚本配置

### 概述
本项目基于 **EasyR1** 框架（veRL 的 fork），针对本地集群环境进行了适配。

### 环境问题
- **PyTorch 版本**：2.9.0+cu128（较新）
- **Flash Attention 2**：无法安装（没有兼容的预编译 wheel）
- **解决方案**：使用 PyTorch 内置的 SDPA 替代

---

### 修改的文件

#### `verl/workers/fsdp_workers.py`
| 行号 | 原始值 | 修改后 |
|------|--------|--------|
| 213 | `attn_implementation="flash_attention_2"` | `attn_implementation="sdpa"` |
| 223 | `attn_implementation="flash_attention_2"` | `attn_implementation="sdpa"` |

**原因**：`flash_attn` 包未安装，torch 2.9.0 没有预编译 wheel

#### `examples/qwen3_think_safety.sh`
- 添加项目根目录自动检测
- 添加 PYTHONPATH 设置
- 添加 conda 环境 Python 自动检测
- 配置 Qwen3-4B-Thinking-2507 模型
- 添加  worker.actor.padding_free=false 

#### `examples/config1.yaml`
- 数据集：`chuhac/R2D-R1`
- Reward function：`reasoning-trace.py:compute_score`
- 单 GPU 配置

---

### 新增的文件

#### `patches/apply_patches.sh`
应用环境适配补丁的脚本

#### `patches/revert_patches.sh`
还原补丁的脚本（当 flash_attn 可用时使用）

#### `examples/run_without_flash_attn.py`
Monkey patch 包装脚本（已弃用）

---

### 运行方式

```bash
# 1. 申请 GPU 资源
salloc --account=p32013 -p gengpu --gres=gpu:a100:2 --cpus-per-task=48 --mem=100G -t 48:00:00

# 2. 激活环境
conda activate vllm

# 3. 运行训练
bash examples/qwen3_think_safety.sh
```

---

### 性能对比

| Attention 实现 | 速度 | 显存 | 安装要求 |
|----------------|------|------|----------|
| flash_attention_2 | 最快 | 最优 | 需额外安装 |
| sdpa (当前使用) | ~95% | ~95% | PyTorch 内置 |

---

### 待办

- [ ] 等待 flash_attn 发布 torch 2.9 兼容版本
- [ ] 测试多 GPU 分布式训练
- [ ] 优化单卡显存使用


## [2025-12-29] - reward function修改

### 修改的文件

#### examples/reward_function/reasoning-trace.py
使用全局缓存，模型只加载一次：
添加全局变量：_CACHED_MODEL, _CACHED_PROJECTION_HEAD, _CACHED_SAFETY_HEAD
首次调用时加载并缓存
后续调用直接使用缓存的模型
使用 torch.float16 和 device_map="auto" 减少内存使用