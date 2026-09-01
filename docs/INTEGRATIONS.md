# 🔌 Integrations — Native Hooks

| Framework | Path | Status | Install |
|-----------|------|--------|---------|
| **PyTorch** | `integrations/pytorch.py` | ✅ Stable | `pip install -e ".[torch]"` |
| **Hugging Face** | `integrations/huggingface.py` | 🚧 Dev | `pip install -e ".[dev]"` |
| **TensorFlow** | `integrations/tensorflow.py` | 🚧 Dev | `pip install -e ".[dev]"` |
| **Albumentations** | `integrations/albumentations.py` | 🔮 Future Work | `pip install -e ".[albumentations]"` (planned) |

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

## 🧪 Albumentations (Future Work)
> Integrate with Albumentations — bridge our smart planner + Vision LLM gates with `A.Compose`.

```python
# 🔮 Future API (planned)
from enhancement_multiagent.planner.augmentation_planner import AugmentationPlanner
from enhancement_multiagent.integrations.albumentations import AlbumentationsAugmenter

plan = AugmentationPlanner().plan_from_hint("low_light")
augmenter = AlbumentationsAugmenter(plan, quality_mode="vision")
compose = augmenter.get_compose()  # albumentations.Compose
augmented = compose(image=image)["image"]
```

See `README.md` → *Future Work — Integrate with Albumentations* for roadmap and `src/enhancement_multiagent/integrations/albumentations.py` stub.

## ⚠️ Dev Note
HF/TF hooks use temp files via `tempfile.gettempdir()` for OpenCV interop and are validated with quality gates. Stable PyTorch hook is recommended for production; HF/TF will graduate after broader testing. Albumentations is 🔮 future — track issue for ETA.
