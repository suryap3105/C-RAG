
import argparse
from crag.data.loaders import MetaQALoader, WebQSPLoader

def audit_leakage():
    """
    Checks if any Test questions appear in Train set.
    For this prototype, we simulate checking partitions.
    """
    print("=== Auditing Dataset Leakage ===")
    
    # 1. Load Data
    loader = MetaQALoader()
    data = loader.load()
    
    # Simulate meaningful split (In real life, load distinct files)
    split_idx = int(len(data) * 0.8)
    train = data[:split_idx]
    test = data[split_idx:]
    
    print(f"Train Size: {len(train)}")
    print(f"Test Size: {len(test)}")
    
    # 2. Check Overlap
    train_qs = set([d['query'].lower().strip() for d in train])
    leakage_count = 0
    
    for item in test:
        q = item['query'].lower().strip()
        if q in train_qs:
            leakage_count += 1
            print(f"[WARN] Leakage found: '{q}'")
            
    if leakage_count == 0:
        print("[PASS] No direct leakage detected between simulated splits.")
    else:
        print(f"[FAIL] {leakage_count} leaking samples found.")

if __name__ == "__main__":
    audit_leakage()
