#!/usr/bin/env python3
"""
Utility to count QA pairs across all downloaded datasets.
"""
import json
import os
from pathlib import Path

def count_squad(path):
    """Count questions in SQuAD format."""
    if not os.path.exists(path):
        return 0
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    count = 0
    for article in data.get('data', []):
        for paragraph in article.get('paragraphs', []):
            count += len(paragraph.get('qas', []))
    return count

def count_webqsp(path):
    """Count questions in WebQSP format."""
    if not os.path.exists(path):
        return 0
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return len(data) if isinstance(data, list) else 0

def count_metaqa_txt(path):
    """Count questions in MetaQA text format."""
    if not os.path.exists(path):
        return 0
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if '\t' in line)

def main():
    base_dir = Path("data")
    
    datasets = {
        "SQuAD v2 Train": count_squad(base_dir / "squad_train_v2.json"),
        "SQuAD v2 Dev": count_squad(base_dir / "squad_dev_v2.json"),
        "WebQSP Test": count_webqsp(base_dir / "webqsp/input/webqsp.examples.test.wikidata.json"),
        "WebQSP Train": count_webqsp(base_dir / "webqsp/input/webqsp.examples.train.json"),
    }
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    total = 0
    for name, count in datasets.items():
        if count > 0:
            print(f"{name:30s}: {count:>10,} questions")
            total += count
    
    print("-"*60)
    print(f"{'TOTAL':30s}: {total:>10,} questions")
    print("="*60 + "\n")
    
    if total >= 100000:
        print(f"[OK] Target reached: {total:,} >= 100,000 questions")
    else:
        print(f"[X] Need {100000 - total:,} more questions to reach 100k target")

if __name__ == "__main__":
    main()
