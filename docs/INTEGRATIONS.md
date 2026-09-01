# 🔌 Integrations — Native Hooks

| Framework | Path | Status | Install |
|-----------|------|--------|---------|
| **PyTorch** | `integrations/pytorch.py` | ✅ Stable | `pip install -e ".[torch]"` |
| **Hugging Face** | `integrations/huggingface.py` | 🚧 Dev | `pip install -e ".[dev]"` |
| **TensorFlow** | `integrations/tensorflow.py` | 🚧 Dev | `pip install -e ".[dev]"` |

> make integrator like huggingface and tensorflow under dev — see `pyproject.toml:21` `dev` extra.

## 🐍 PyTorch
```python
from enhancement_multiagent.planner.augmentation_planner import AugmentationPlanner
from enhancement_multiagent.integrations.pytorch import SmartAugmentationDataset

plan = AugmentationPlanner().plan_from_hint("low_light")
smart_ds = SmartAugmentationDataset(base_dataset, plan, quality_mode="vision", apply_prob=0.7)
# or transform: from integrations.pytorch import get_pytorch_transform
```

## 🤗 Hugging Face (Dev)
```python
from enhancement_multiagent.integrations.huggingface import HFDatasetAugmenter
aug = HFDatasetAugmenter(plan)
ds_aug = aug.augment_dataset(dataset, image_column="image", num_augmentations=2)
```

## 🔷 TensorFlow (Dev)
```python
from enhancement_multiagent.integrations.tensorflow import get_tf_augmentation_layer
layer = get_tf_augmentation_layer(plan)
ds = ds.map(lambda x,y: (layer(x), y))
```

## ⚠️ Dev Note
HF/TF hooks use temp files via `tempfile.gettempdir()` for OpenCV interop and are validated with quality gates. Stable PyTorch hook is recommended for production; HF/TF will graduate after broader testing.
