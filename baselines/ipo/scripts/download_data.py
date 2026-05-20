"""
Download the STAR-1 dataset.
Paper Training Settings (Appendix A.2): uses 1,000 harmful prompts + 915 benign prompts from STAR-1.

STAR-1 dataset sources:
- Paper: STAR-1: Safer Alignment of Reasoning LLMs with 1K Data (Wang et al., 2025)
- Project page: https://ucsc-vlaa.github.io/STAR-1/
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
    Download the STAR-1 dataset from HuggingFace.
    Paper Appendix A.2: https://huggingface.co/datasets/UCSC-VLAA/STAR-1
    STAR-1 is entirely harmful; fields: id, question, response, category, source, score.
    """
    if not HAS_DATASETS:
        print("Error: datasets library required: pip install datasets")
        return None

    print("Downloading STAR-1 dataset from HuggingFace...")
    print("Dataset URL: https://huggingface.co/datasets/UCSC-VLAA/STAR-1")
    
    try:
        # Load STAR-1 dataset
        dataset = load_dataset("UCSC-VLAA/STAR-1")
        
        star1_data = []
        
        # STAR-1 is entirely harmful; keep all fields
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
        print(f"Download failed: {e}")
        print("Please download manually: https://huggingface.co/datasets/UCSC-VLAA/STAR-1")
        return None


def download_star_benign_915_from_huggingface():
    """
    Download the STAR-benign-915 dataset from HuggingFace.
    Dataset URL: https://huggingface.co/datasets/UCSC-VLAA/STAR-benign-915
    """
    if not HAS_DATASETS:
        print("Error: datasets library required: pip install datasets")
        return None

    print("Downloading STAR-benign-915 dataset from HuggingFace...")
    print("Dataset URL: https://huggingface.co/datasets/UCSC-VLAA/STAR-benign-915")
    
    try:
        # Load STAR-benign-915 dataset
        dataset = load_dataset("UCSC-VLAA/STAR-benign-915")
        
        benign_915_prompts = []
        
        # STAR-benign-915 contains 915 benign prompts
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
        print(f"Download failed: {e}")
        print("Please download manually: https://huggingface.co/datasets/UCSC-VLAA/STAR-benign-915")
        return None


def save_datasets(output_dir: str):
    """Save datasets to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download and save the STAR-1 dataset
    star1_data = download_star1_from_huggingface()
    
    if star1_data is not None:
        star1_path = output_path / "star1.json"
        with open(star1_path, "w", encoding="utf-8") as f:
            json.dump(star1_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(star1_data)} STAR-1 examples to {star1_path}")
    else:
        print("STAR-1 download failed; please download manually")

    # Download and save the STAR-benign-915 dataset
    print()
    benign_915 = download_star_benign_915_from_huggingface()

    if benign_915 is not None:
        benign_915_path = output_path / "star_benign_915.json"
        with open(benign_915_path, "w", encoding="utf-8") as f:
            json.dump(benign_915, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(benign_915)} STAR-benign-915 prompts to {benign_915_path}")
    else:
        print("STAR-benign-915 download failed; please download manually")


def main():
    print("=" * 60)
    print("Downloading Datasets")
    print("  - STAR-1: Safer Alignment of Reasoning LLMs with 1K Data")
    print("  - STAR-benign-915: 915 benign prompts")
    print("=" * 60)
    print()
    
    output_dir = Path(__file__).parent.parent / "data"
    save_datasets(str(output_dir))
    
    print()
    print("Dataset download complete!")
    # print()
    # print("如果下载失败，请手动下载:")
    # print("  - STAR-1: https://huggingface.co/datasets/UCSC-VLAA/STAR-1")
    # print("  - STAR-benign-915: https://huggingface.co/datasets/UCSC-VLAA/STAR-benign-915")
    # print("  # 确保已安装: pip install datasets")


if __name__ == "__main__":
    main()
