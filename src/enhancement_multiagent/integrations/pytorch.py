"""PyTorch integration - native hook for smart augmentation."""
from typing import Callable, List, Dict, Any, Optional
import os
import tempfile
import cv2
import numpy as np


class SmartAugmentationDataset:
    """
    Wrap any PyTorch dataset to apply model-aware augmentations.
    Example:
        base_ds = MyDataset(...)
        planner = AugmentationPlanner()
        plan = planner.plan_from_hint("low_light")
        smart_ds = SmartAugmentationDataset(base_ds, plan, quality_gate="vision")
    Falls back gracefully if torch not installed (still importable for inspection).
    """

    def __init__(self, base_dataset, augmentation_plan: List[Dict[str, Any]], quality_mode: str = "all", apply_prob: float = 0.7):
        self.base = base_dataset
        self.plan = augmentation_plan
        self.quality_mode = quality_mode
        self.apply_prob = apply_prob
        try:
            from ..quality.vision_llm import QualityOrchestrator
            self.quality = QualityOrchestrator()
        except Exception:
            self.quality = None
        try:
            from ..agents.orchestrator import MultiAgentOrchestrator
            self.orchestrator = MultiAgentOrchestrator()
        except Exception:
            self.orchestrator = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        # Detect image format: assume (image, label) or dict with 'image'
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            image, label = item[0], item[1]
            rest = item[2:] if len(item) > 2 else ()
        elif isinstance(item, dict) and "image" in item:
            image = item["image"]
            label = item.get("label")
            rest = ()
        else:
            return item

        # If image is numpy, apply augmentation via temp files (to reuse agent pipeline)
        # For tensor, convert to numpy
        try:
            import torch
            is_tensor = isinstance(image, torch.Tensor)
        except ImportError:
            is_tensor = False

        # Only augment with prob
        import random
        if random.random() > self.apply_prob or self.orchestrator is None:
            return item

        # Convert to numpy BGR for OpenCV pipeline if needed
        # Assume image is HWC or CHW tensor normalized? Keep simple: if tensor, denormalize naively
        tmpdir = tempfile.gettempdir()
        tmp_in = os.path.join(tmpdir, f"_smart_aug_in_{idx}.png")
        tmp_out = os.path.join(tmpdir, f"_smart_aug_out_{idx}.png")
        try:
            # Handle tensor -> numpy
            if is_tensor:
                # expect shape CxHxW or HxWxC, values 0-1 or 0-255
                import torch
                arr = image.cpu().numpy() if isinstance(image, torch.Tensor) else np.array(image)
                if arr.ndim == 3 and arr.shape[0] in (1,3,4):  # CHW
                    arr = np.transpose(arr, (1,2,0))
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
                if arr.shape[2] == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                cv2.imwrite(tmp_in, arr)
            elif isinstance(image, np.ndarray):
                # assume BGR or RGB? if HWC
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
                cv2.imwrite(tmp_in, image)
            elif isinstance(image, str) and os.path.exists(image):
                tmp_in = image
            else:
                return item

            # Pick one step from plan randomly
            step = random.choice(self.plan) if self.plan else {"agent": "geometric", "operation": "flip", "params": {"direction": "horizontal"}}
            res = self.orchestrator.execute(tmp_in, tmp_out, step["agent"], step["operation"], **step.get("params", {}))
            if not res.success:
                return item
            # Quality check
            if self.quality:
                q = self.quality.validate(tmp_in, tmp_out, mode=self.quality_mode)
                if not q["final_pass"]:
                    return item  # return original if augmentation fails quality
            # Load augmented back
            aug = cv2.imread(tmp_out)
            if aug is None:
                return item
            if is_tensor:
                aug_rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
                # Convert back to tensor shape CHW 0-1
                import torch
                aug_t = torch.from_numpy(aug_rgb).permute(2,0,1).float() / 255.0
                if isinstance(item, (tuple,list)):
                    return (aug_t, label, *rest)
                else:
                    new_item = dict(item)
                    new_item["image"] = aug_t
                    return new_item
            else:
                # Return numpy
                if isinstance(item, (tuple,list)):
                    return (aug, label, *rest)
                else:
                    new_item = dict(item)
                    new_item["image"] = aug
                    return new_item
        except Exception:
            return item
        finally:
            for p in [tmp_in, tmp_out]:
                try:
                    if os.path.exists(p) and "_smart_aug" in p:
                        os.remove(p)
                except Exception:
                    pass


def get_pytorch_transform(augmentation_plan: List[Dict[str, Any]]):
    """Return a torchvision-compatible transform callable."""
    def transform(image):
        # image is PIL or numpy
        try:
            import numpy as np
            import tempfile, os
            from PIL import Image
            if isinstance(image, Image.Image):
                arr = np.array(image)
                arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                arr_bgr = image
            tmpdir = tempfile.gettempdir()
            tmp_in = os.path.join(tmpdir, "_torch_transform_in.png")
            tmp_out = os.path.join(tmpdir, "_torch_transform_out.png")
            cv2.imwrite(tmp_in, arr_bgr)
            from ..agents.orchestrator import MultiAgentOrchestrator
            orchestrator = MultiAgentOrchestrator()
            import random
            step = random.choice(augmentation_plan)
            orchestrator.execute(tmp_in, tmp_out, step["agent"], step["operation"], **step.get("params", {}))
            aug = cv2.imread(tmp_out)
            aug_rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
            return Image.fromarray(aug_rgb)
        except Exception:
            return image
    return transform
