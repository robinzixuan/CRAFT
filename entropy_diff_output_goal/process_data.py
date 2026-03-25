import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 路径配置
diff_path = "token_entropy_diff.csv"  # 请替换为你的实际路径
output_csv = "token_entropy_diff_labeled.csv"
histogram_path = "token_entropy_diff_histogram.png"

# 读取数据
df = pd.read_csv(diff_path)

# 确保列存在
assert 'token' in df.columns and 'avg_entropy_diff' in df.columns, "Missing required columns"

# 计算分位数
q50 = df['avg_entropy_diff'].quantile(0.5)
q90 = df['avg_entropy_diff'].quantile(0.9)
q95 = df['avg_entropy_diff'].quantile(0.95)
q99 = df['avg_entropy_diff'].quantile(0.99)

print(f"Median (50th): {q50:.4f}, 90th: {q90:.4f}, 95th: {q95:.4f}, 99th: {q99:.4f}")

# 分类函数
def categorize(diff):
    if diff >= q95:
        return "high"
    elif diff <= q50:
        return "low"
    else:
        return "medium"

# 添加分类列
df["diff_category"] = df["avg_entropy_diff"].apply(categorize)

# 保存标注后的 CSV
df.to_csv(output_csv, index=False)
print(f"[✔] 输出已保存到 {output_csv}")

# 可选：绘图
plt.figure(figsize=(10, 5))
plt.hist(df["avg_entropy_diff"], bins=100, color='lightcoral', edgecolor='black')
plt.axvline(q50, color='blue', linestyle='--', label='50th (median)')
plt.axvline(q95, color='green', linestyle='--', label='95th')
plt.axvline(q99, color='red', linestyle='--', label='99th')
plt.title("Token Avg Entropy Diff Distribution")
plt.xlabel("Avg Entropy Diff")
plt.ylabel("Token Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(histogram_path)
print(f"[✔] 直方图已保存到 {histogram_path}")
