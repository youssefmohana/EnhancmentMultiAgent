"""TensorFlow / Keras integration."""
from typing import List, Dict, Any
import os
import tempfile
import cv2
import numpy as np


def get_tf_augmentation_layer(augmentation_plan: List[Dict[str, Any]]):
    """
    Returns a Keras layer that applies smart augmentations via tf.numpy_function.
    If TensorFlow not installed, returns a no-op callable with same signature.
    Usage:
        layer = get_tf_augmentation_layer(plan)
        dataset = dataset.map(lambda x,y: (layer(x), y))
    """
    try:
        import tensorflow as tf

        # Build python function that TF will wrap
        from ..agents.orchestrator import MultiAgentOrchestrator
        from ..quality.vision_llm import QualityOrchestrator

        orchestrator = MultiAgentOrchestrator()
        quality = QualityOrchestrator()

        def _aug_fn(image_np):
            # image_np is numpy HWC uint8 or float
            tmp_in = os.path.join(tempfile.gettempdir(), "_tf_aug_in.png")
            tmp_out = os.path.join(tempfile.gettempdir(), "_tf_aug_out.png")
            try:
                if image_np.dtype != np.uint8:
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                    else:
                        image_np = image_np.astype(np.uint8)
                # TF uses RGB, OpenCV BGR
                bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(tmp_in, bgr)
                import random
                step = random.choice(augmentation_plan) if augmentation_plan else {"agent": "geometric", "operation": "flip", "params": {"direction": "horizontal"}}
                res = orchestrator.execute(tmp_in, tmp_out, step["agent"], step["operation"], **step.get("params", {}))
                if not res.success:
                    return image_np
                q = quality.validate(tmp_in, tmp_out, mode="classical")
                if not q["final_pass"]:
                    return image_np
                aug_bgr = cv2.imread(tmp_out)
                aug_rgb = cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2RGB)
                return aug_rgb
            except Exception:
                return image_np
            finally:
                for p in [tmp_in, tmp_out]:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass

        @tf.function
        def tf_augment(image, label=None):
            # image: tf.Tensor HWC
            aug = tf.numpy_function(_aug_fn, [image], tf.uint8)
            aug.set_shape(image.shape)
            if label is not None:
                return aug, label
            return aug

        return tf_augment

    except ImportError:
        # No TF available: return identity
        def no_tf(image, label=None):
            if label is not None:
                return image, label
            return image
        no_tf.is_noop = True
        return no_tf


class TFDatasetAugmenter:
    """Helper to wrap tf.data.Dataset."""

    def __init__(self, augmentation_plan: List[Dict[str, Any]], apply_prob: float = 0.6):
        self.plan = augmentation_plan
        self.prob = apply_prob
        self._layer = get_tf_augmentation_layer(augmentation_plan)

    def augment(self, dataset):
        """Apply augmentation to dataset with probability."""
        try:
            import tensorflow as tf
            import random

            def maybe_augment(image, label):
                # Random gate outside tf.numpy_function for prob
                if random.random() > self.prob:
                    return image, label
                return self._layer(image, label)

            return dataset.map(maybe_augment, num_parallel_calls=tf.data.AUTOTUNE)
        except ImportError:
            return dataset
