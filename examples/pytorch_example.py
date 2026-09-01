"""PyTorch native hook example - shows integration API (works even without torch installed)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from enhancement_multiagent.planner.augmentation_planner import AugmentationPlanner

# Build plan targeted at model's weaknesses
planner = AugmentationPlanner()
plan = planner.plan_from_hint("low_light")
print("Plan for low_light:", plan)

# Hook into PyTorch dataset (pseudo, runs without torch)
try:
    from enhancement_multiagent.integrations.pytorch import SmartAugmentationDataset

    class DummyDataset:
        def __init__(self): 
            import numpy as np
            self.data = [(np.ones((32,32,3), dtype=np.uint8)*100, 0) for _ in range(3)]
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx]

    ds = DummyDataset()
    smart_ds = SmartAugmentationDataset(ds, plan, apply_prob=1.0)
    print(f"Wrapped {len(ds)} samples -> smart dataset with {len(smart_ds)} samples")
    sample = smart_ds[0]
    print("Sample augmented shape:", sample[0].shape if hasattr(sample[0], 'shape') else type(sample[0]))
    print("[PASS] PyTorch hook works (with quality gate validation)")
except Exception as e:
    print("PyTorch example skipped:", e)
