import pandas as pd
import numpy as np
import os
from collections import defaultdict

# 输入路径
instruct_csv = "/projects/p32013/neurons/entropy_results_instruct/advbench_target/token_entropy.csv"
vanilla_csv = "/projects/p32013/neurons/entropy_vanilla/advbench_target/token_entropy.csv"

# 输出路径
output_dir = "./entropy_diff_results_target"
os.makedirs(output_dir, exist_ok=True)
diff_csv_path = os.path.join(output_dir, "token_entropy_diff.csv")
top_diff_txt_path = os.path.join(output_dir, "top_token_entropy_diff.txt")

# 读取 CSV
df_instruct = pd.read_csv(instruct_csv)
df_vanilla = pd.read_csv(vanilla_csv)

# 对齐：按 token、text_index、position_in_text
key_cols = ["token", "text_index", "position_in_text"]
merged = pd.merge(df_instruct, df_vanilla, on=key_cols, suffixes=("_instruct", "_vanilla"))

# 计算 entropy 差值
merged["entropy_diff"] = merged["entropy_instruct"] - merged["entropy_vanilla"]

# 保存差值 CSV
merged.to_csv(diff_csv_path, index=False)

# 选取前 20% 差值 token
k = int(len(merged) * 0.20)
top_k = merged.nlargest(k, "entropy_diff")

# 统计频率与平均差值
token_stats = defaultdict(lambda: {"count": 0, "total_diff": 0.0})
for _, row in top_k.iterrows():
    token = row["token"]
    token_stats[token]["count"] += 1
    token_stats[token]["total_diff"] += row["entropy_diff"]

# 生成排序列表：按频率排序
sorted_stats = sorted(token_stats.items(), key=lambda x: x[1]["count"], reverse=True)

# 写入 TXT 文件
with open(top_diff_txt_path, "w", encoding="utf-8") as f:
    f.write("Top 100 High Entropy Diff Tokens and their Frequencies + Mean Diff:\n")
    for i, (token, stats) in enumerate(sorted_stats[:100], 1):
        mean_diff = stats["total_diff"] / stats["count"]
        f.write(f"{i}. Token: '{token}', Frequency: {stats['count']}, Mean Entropy Diff: {mean_diff:.4f}\n")

print(f"✓ 差值 CSV 已保存：{diff_csv_path}")
print(f"✓ 含差值的高差异 token 列表已保存：{top_diff_txt_path}")
