#!/usr/bin/env python3
"""Legacy shim — use `src/enhancement_multiagent/pipelines/augmentation.py` instead."""
import warnings
warnings.warn("augmentation_pipeline.py is deprecated, use enhancement_multiagent.pipelines.augmentation", DeprecationWarning, stacklevel=2)
from enhancement_multiagent.pipelines.augmentation import run_augmentation_pipeline, main  # noqa: F401
if __name__ == "__main__":
    main()
