import pandas as pd
import os

# 所有输入目录列表（每个目录下都有一个 token_entropy.csv）
input_dirs = [
    "/projects/p32013/neurons/entropy_results_instruct/advbench_goal",
    "/projects/p32013/neurons/entropy_results_instruct/advbench_target",
    "/projects/p32013/neurons/entropy_vanilla/advbench_goal",
    "/projects/p32013/neurons/entropy_vanilla/advbench_target",
    # 添加其他目录路径
]
top_pct = 0.20  # 提取前 20% 高平均 entropy 的唯一 token

for input_dir in input_dirs:
    input_csv = os.path.join(input_dir, "token_entropy.csv")
    
    if not os.path.exists(input_csv):
        print(f"[!] Missing file: {input_csv}")
        continue

    df = pd.read_csv(input_csv)

    # 按 token 分组，计算平均 entropy 和出现次数
    grouped = df.groupby("token").agg(
        avg_entropy=("entropy", "mean"),
        count=("entropy", "count"),
    ).reset_index()

    # 按 avg_entropy 排序，提取前 20%
    grouped_sorted = grouped.sort_values(by="avg_entropy", ascending=False)
    k = int(len(grouped_sorted) * top_pct)
    top_tokens = grouped_sorted.head(k)

    # 输出路径
    csv_output = os.path.join(input_dir, "top_20_percent_unique_avg_entropy_tokens.csv")
    txt_output = os.path.join(input_dir, "top_20_percent_unique_avg_entropy_tokens.txt")

    # # 保存为 CSV
    # top_tokens.to_csv(csv_output, index=False)

    # 保存为 TXT
    with open(txt_output, "w", encoding="utf-8") as f:
        f.write("Top 20% Unique Tokens by Avg Entropy:\n\n")
        for i, row in top_tokens.iterrows():
            f.write(f"{i+1}. Token: '{row['token']}', Avg Entropy: {row['avg_entropy']:.6f}, Count: {row['count']}\n")

    print(f"[✓] Processed: {input_csv}")