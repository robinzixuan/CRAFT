import pandas as pd
import numpy as np
from collections import defaultdict
import os

# 读取两个模型的 token_entropy.csv
instruct_path = "/projects/p32013/neurons/entropy_results_instruct/advbench_target/token_entropy.csv"
vanilla_path = "/projects/p32013/neurons/entropy_vanilla/advbench_target/token_entropy.csv"
out_dir = "./entropy_diff_output_target"
os.makedirs(out_dir, exist_ok=True)

df1 = pd.read_csv(instruct_path)
df2 = pd.read_csv(vanilla_path)

# 只保留 token 和 entropy 两列
df1 = df1[['token', 'entropy']].copy()
df2 = df2[['token', 'entropy']].copy()

# 合并两表：同一 token 可能出现多次
group1 = df1.groupby('token')['entropy'].apply(list)
group2 = df2.groupby('token')['entropy'].apply(list)

# 对相同 token 计算平均差值
common_tokens = set(group1.index).intersection(set(group2.index))
token_stats = defaultdict(lambda: {"count": 0, "total_diff": 0.0})

for token in common_tokens:
    entropies_1 = group1[token]
    entropies_2 = group2[token]
    min_len = min(len(entropies_1), len(entropies_2))
    diffs = np.abs(np.array(entropies_1[:min_len]) - np.array(entropies_2[:min_len]))
    token_stats[token]["count"] = min_len
    token_stats[token]["total_diff"] = diffs.sum()

# 改为按平均差值排序（从大到小）
sorted_stats = sorted(token_stats.items(), key=lambda x: x[1]["total_diff"] / x[1]["count"], reverse=True)

# 输出 token-level 差值 CSV
rows = []
for token, stats in sorted_stats:
    avg_diff = stats["total_diff"] / stats["count"]
    rows.append({"token": token, "count": stats["count"], "avg_entropy_diff": avg_diff})
df_diff = pd.DataFrame(rows)
df_diff.to_csv(os.path.join(out_dir, "token_entropy_diff.csv"), index=False)

# 写 Top 20% 的 TXT 文件
top_k = int(0.20 * len(sorted_stats))
with open(os.path.join(out_dir, "top_20_percent_tokens_by_entropy_diff.txt"), "w", encoding="utf-8") as f:
    f.write("Top Tokens by Average Entropy Difference:\n")
    for i, (token, stats) in enumerate(sorted_stats[:top_k]):
        avg_diff = stats["total_diff"] / stats["count"]
        f.write(f"{i+1}. Token: '{token}', Frequency: {stats['count']}, Avg_Diff: {avg_diff:.4f}\n")

print(f"输出已保存到：{out_dir}")
