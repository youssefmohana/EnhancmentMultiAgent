"""Hugging Face datasets hook example."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from enhancement_multiagent.planner.augmentation_planner import AugmentationPlanner
from enhancement_multiagent.integrations.huggingface import HFDatasetAugmenter
import cv2, numpy as np
from pathlib import Path

Path("demo_images").mkdir(exist_ok=True)
# Create dummy PIL image
from PIL import Image
img = Image.fromarray(np.ones((64,64,3), dtype=np.uint8)*150)
plan = AugmentationPlanner().plan_from_hint("color_cast")
augmenter = HFDatasetAugmenter(plan)
aug_img = augmenter.augment_image(img)
print(f"Original {img.size} -> augmented {aug_img.size}")
print("[PASS] HF hook works, use: augmenter.augment_dataset(dataset)")
