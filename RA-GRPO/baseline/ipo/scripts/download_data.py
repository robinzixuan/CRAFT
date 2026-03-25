"""
下载STAR-1数据集
论文Training Settings (Appendix A.2): 使用STAR-1的1000个有害提示 + 915个良性提示

STAR-1数据集来源:
- 论文: STAR-1: Safer Alignment of Reasoning LLMs with 1K Data (Wang et al., 2025)
- 官方页面: https://ucsc-vlaa.github.io/STAR-1/
- GitHub: https://github.com/ucsc-vlaa/STAR-1
- HuggingFace: https://huggingface.co/datasets/UCSC-VLAA/STAR-1
"""

import json
import os
from pathlib import Path

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def download_star1_from_huggingface():
    """
    从HuggingFace下载STAR-1数据集
    论文附录A.2: https://huggingface.co/datasets/UCSC-VLAA/STAR-1
    STAR-1数据集都是harmful的，包含字段: id, question, response, category, source, score
    """
    if not HAS_DATASETS:
        print("Error: 需要安装datasets库: pip install datasets")
        return None
    
    print("从HuggingFace下载STAR-1数据集...")
    print("数据集地址: https://huggingface.co/datasets/UCSC-VLAA/STAR-1")
    
    try:
        # 加载STAR-1数据集
        dataset = load_dataset("UCSC-VLAA/STAR-1")
        
        star1_data = []
        
        # STAR-1数据集都是harmful的，包含所有字段
        for item in dataset['train']:
            star1_data.append({
                "id": item.get('id', ''),
                "question": item.get('question', ''),
                "response": item.get('response', ''),
                "category": item.get('category', []),
                "source": item.get('source', ''),
                "score": item.get('score', {}),
            })
        
        return star1_data
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载: https://huggingface.co/datasets/UCSC-VLAA/STAR-1")
        return None


def download_star_benign_915_from_huggingface():
    """
    从HuggingFace下载STAR-benign-915数据集
    数据集地址: https://huggingface.co/datasets/UCSC-VLAA/STAR-benign-915
    """
    if not HAS_DATASETS:
        print("Error: 需要安装datasets库: pip install datasets")
        return None
    
    print("从HuggingFace下载STAR-benign-915数据集...")
    print("数据集地址: https://huggingface.co/datasets/UCSC-VLAA/STAR-benign-915")
    
    try:
        # 加载STAR-benign-915数据集
        dataset = load_dataset("UCSC-VLAA/STAR-benign-915")
        
        benign_915_prompts = []
        
        # STAR-benign-915数据集包含915个良性提示
        for item in dataset['train']:
            benign_915_prompts.append({
                "id": item.get('id', ''),
                "question": item.get('question', ''),
                "response": item.get('response', ''),
                "source": item.get('source', ''),
                "score": item.get('score', {}),
            })
        
        return benign_915_prompts
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载: https://huggingface.co/datasets/UCSC-VLAA/STAR-benign-915")
        return None


def save_datasets(output_dir: str):
    """保存数据集"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 下载并保存STAR-1数据集
    star1_data = download_star1_from_huggingface()
    
    if star1_data is not None:
        star1_path = output_path / "star1.json"
        with open(star1_path, "w", encoding="utf-8") as f:
            json.dump(star1_data, f, indent=2, ensure_ascii=False)
        print(f"保存了 {len(star1_data)} 个STAR-1数据到 {star1_path}")
    else:
        print("STAR-1数据集下载失败，请手动下载")
    
    # 下载并保存STAR-benign-915数据集
    print()
    benign_915 = download_star_benign_915_from_huggingface()
    
    if benign_915 is not None:
        benign_915_path = output_path / "star_benign_915.json"
        with open(benign_915_path, "w", encoding="utf-8") as f:
            json.dump(benign_915, f, indent=2, ensure_ascii=False)
        print(f"保存了 {len(benign_915)} 个STAR-benign-915提示到 {benign_915_path}")
    else:
        print("STAR-benign-915数据集下载失败，请手动下载")


def main():
    print("=" * 60)
    print("下载数据集")
    print("  - STAR-1数据集: Safer Alignment of Reasoning LLMs with 1K Data")
    print("  - STAR-benign-915数据集: 915个良性提示")
    print("=" * 60)
    print()
    
    output_dir = Path(__file__).parent.parent / "data"
    save_datasets(str(output_dir))
    
    print()
    print("数据集下载完成!")
    # print()
    # print("如果下载失败，请手动下载:")
    # print("  - STAR-1: https://huggingface.co/datasets/UCSC-VLAA/STAR-1")
    # print("  - STAR-benign-915: https://huggingface.co/datasets/UCSC-VLAA/STAR-benign-915")
    # print("  # 确保已安装: pip install datasets")


if __name__ == "__main__":
    main()
