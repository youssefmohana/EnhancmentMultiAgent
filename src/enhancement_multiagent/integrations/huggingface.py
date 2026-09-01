"""Hugging Face datasets integration."""
from typing import List, Dict, Any
import os
import tempfile
import cv2
import random


class HFDatasetAugmenter:
    """
    Native hook for Hugging Face datasets.
    Example:
        from datasets import load_dataset
        ds = load_dataset("cifar10", split="train")
        augmenter = HFDatasetAugmenter(plan)
        ds_aug = augmenter.augment_dataset(ds, image_column="img", num_augmentations=2)
    """

    def __init__(self, augmentation_plan: List[Dict[str, Any]], quality_mode: str = "all"):
        self.plan = augmentation_plan
        self.quality_mode = quality_mode
        try:
            from ..agents.orchestrator import MultiAgentOrchestrator
            from ..quality.vision_llm import QualityOrchestrator
            self.orchestrator = MultiAgentOrchestrator()
            self.quality = QualityOrchestrator()
        except Exception:
            self.orchestrator = None
            self.quality = None

    def augment_image(self, image):
        """Augment a single PIL/numpy image. Returns augmented PIL image."""
        try:
            from PIL import Image
            import numpy as np
            if isinstance(image, Image.Image):
                arr = np.array(image)
                is_pil = True
            elif isinstance(image, np.ndarray):
                arr = image
                is_pil = False
            else:
                return image

            # Convert RGB -> BGR for OpenCV
            if arr.ndim == 3 and arr.shape[2] == 3:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                bgr = arr

            tmpdir = tempfile.gettempdir()
            tmp_in = os.path.join(tmpdir, "_hf_aug_in.png")
            tmp_out = os.path.join(tmpdir, "_hf_aug_out.png")
            cv2.imwrite(tmp_in, bgr)
            step = random.choice(self.plan) if self.plan else {"agent": "geometric", "operation": "flip", "params": {"direction": "horizontal"}}
            if self.orchestrator is None:
                return image
            res = self.orchestrator.execute(tmp_in, tmp_out, step["agent"], step["operation"], **step.get("params", {}))
            if not res.success:
                return image
            if self.quality:
                q = self.quality.validate(tmp_in, tmp_out, mode=self.quality_mode)
                if not q["final_pass"]:
                    return image
            aug_bgr = cv2.imread(tmp_out)
            aug_rgb = cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2RGB)
            if is_pil:
                return Image.fromarray(aug_rgb)
            return aug_rgb
        except Exception:
            return image
        finally:
            tmpdir = tempfile.gettempdir()
            for p in [os.path.join(tmpdir, "_hf_aug_in.png"), os.path.join(tmpdir, "_hf_aug_out.png")]:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

    def augment_dataset(self, dataset, image_column: str = "image", num_augmentations: int = 1, new_column: str = None):
        """
        Augment a HF Dataset. If HF not installed, works with list of dicts.
        Returns augmented dataset (original +augmented copies).
        """
        # Try HF datasets map
        try:
            def _map_fn(example):
                img = example[image_column]
                # Handle batched vs single: assume single
                aug = self.augment_image(img)
                if new_column:
                    example[new_column] = aug
                else:
                    # Keep original, add augmented as new row via augmentation? For simplicity, replace with augmentation probabilistically
                    example[image_column] = aug
                return example

            # If dataset is HF Dataset, use .map
            if hasattr(dataset, "map"):
                augmented = dataset.map(_map_fn)
                if num_augmentations > 1:
                    # Concatenate multiple augmented versions
                    from datasets import concatenate_datasets
                    all_ds = [dataset]
                    for _ in range(num_augmentations):
                        # Fresh random per map
                        all_ds.append(dataset.map(_map_fn))
                    return concatenate_datasets(all_ds)
                return augmented
            # Fallback: list of dicts
            if isinstance(dataset, list):
                result = list(dataset)
                for _ in range(num_augmentations):
                    for item in dataset:
                        new_item = dict(item)
                        new_item[image_column] = self.augment_image(item[image_column])
                        result.append(new_item)
                return result
        except ImportError:
            pass
        return dataset

    def get_transform(self):
        """Return callable for usage as custom transform."""
        return lambda img: self.augment_image(img)
