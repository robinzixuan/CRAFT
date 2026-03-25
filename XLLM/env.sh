#!/usr/bin/env bash
# sync_env.sh  ——  从 SRC_ENV 把“缺少的包”补到 DST_ENV，且不改动 DST_ENV 的 torch/torchvision/torchaudio/triton/numpy

set -euo pipefail

# ===== 配置：源环境与目标环境 =====
SRC_ENV="${SRC_ENV:-jailbreakbench}"   # 源：已有较多包的环境
DST_ENV="${DST_ENV:-mpt}"              # 目标：需要补齐包的环境
# =================================

echo "[0/7] 准备 conda 激活能力..."
if ! command -v conda >/dev/null 2>&1; then
  echo "错误：找不到 conda，请先安装或在含 conda 的 shell 中运行。" >&2
  exit 1
fi
eval "$(conda shell.bash hook)"

# 1) 导出 SRC 的包
echo "[1/7] 导出 ${SRC_ENV} 的包列表..."
conda activate "${SRC_ENV}"
pip list --format=freeze > /tmp/src_freeze.txt

# 2) 导出 DST 的包
echo "[2/7] 导出 ${DST_ENV} 的包列表..."
conda activate "${DST_ENV}"
pip list --format=freeze > /tmp/dst_freeze.txt

# 3) 生成约束：锁住 DST 的关键组合，防止被升级
echo "[3/7] 生成 ${DST_ENV} 的关键版本约束 constraints.txt..."
python - <<'PY'
import importlib, pathlib
keep = ["torch","torchvision","torchaudio","triton","numpy"]
pins = {}
for name in keep:
    try:
        m = importlib.import_module(name)
        pins[name] = getattr(m,"__version__","")
    except Exception:
        pass
text = []
for k,v in pins.items():
    if v:
        text.append(f"{k}=={v}")
# 保险：避免 numpy 被升到 2.x
if "numpy" not in pins:
    text.append("numpy<2")
path = pathlib.Path("/tmp/constraints.txt")
path.write_text("\n".join(text) + ("\n" if text else ""))
print("=== constraints.txt ===")
print(path.read_text() or "(empty)")
PY

# 4) 计算差集 + 过滤可能冲突的大轮子
echo "[4/7] 计算差集并过滤可能与 CUDA/Torch 紧耦合的包..."
# 去掉 pip/setuptools/wheel 自身
grep -v -E '^(pip|setuptools|wheel)=' /tmp/src_freeze.txt > /tmp/src_freeze_clean.txt || true
# 目标已有的包名集合
awk -F'[= ]' '{print tolower($1)}' /tmp/dst_freeze.txt | sort -u > /tmp/dst_names.txt

python - <<'PY'
import re, pathlib
src = pathlib.Path("/tmp/src_freeze_clean.txt").read_text().splitlines()
dst_names = set(pathlib.Path("/tmp/dst_names.txt").read_text().splitlines())

# 黑名单：不跨环境同步这些（容易把 CUDA/Torch 组合拉乱）
black = re.compile(
    r'^(torch|torchvision|torchaudio|triton|numpy|'
    r'cupy|xformers|vllm|pytorch|nvidia-|cuda-|cudnn|cublas|cufft|curand|cusolver|'
    r'cusparse|cusparselt|nccl|nvtx|nvjitlink)', re.I)

missing = []
for line in src:
    # 名称在前，版本约束在后
    if " @ " in line:
        name = line.split("@",1)[0].split("==")[0].split(">=")[0].split("<=")[0].strip().lower()
    else:
        name = re.split(r'[=<>! ]+', line)[0].strip().lower()
    if not name:
        continue
    if name in dst_names:
        continue
    if black.match(name):
        continue
    missing.append(line)

pathlib.Path("/tmp/missing_from_src.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
print(f"需要补装的包数量（过滤后）: {len(missing)}")
if missing:
    print("示例预览:", missing[:20])
PY

# 5) 预览待安装列表
echo "[5/7] 将要安装的包（前 50 行预览）:"
if [ -s /tmp/missing_from_src.txt ]; then
  head -n 50 /tmp/missing_from_src.txt
else
  echo "(空) 没有需要补装的包。"
fi

# 6) 按 constraints 逐个安装缺失包；单个失败不会中断全部
echo "[6/7] 开始安装（逐个执行，失败会跳过并继续）..."
if [ -s /tmp/missing_from_src.txt ]; then
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    echo ">>> pip install -c /tmp/constraints.txt \"$line\""
    if ! pip install -c /tmp/constraints.txt "$line"; then
      echo "!!! 安装失败（已跳过）：$line"
    fi
  done < /tmp/missing_from_src.txt
else
  echo "没有需要安装的条目。"
fi

# 7) 完成
echo "[7/7] 同步完成。当前环境为：${DST_ENV}"
echo "你可以继续在 ${DST_ENV} 里使用已补齐的包。"
