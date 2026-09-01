"""Augmentation Planner - model-aware strategy selection."""

from typing import List, Dict, Any
import random
from .weakness_analyzer import ModelWeaknessAnalyzer


class AugmentationPlanner:
    """
    The Future: A Data Augmentation Planner
    - Analyse model weaknesses
    - Select right agents to target gaps
    - Generate augmentations
    - Validate with Vision LLM quality checks
    - Close loop - feed back and refine strategy
    """

    # Strategy library: maps weakness types to augmentation recipes
    STRATEGY_LIBRARY: Dict[str, List[Dict[str, Any]]] = {
        "low_light": [
            {"agent": "photometric", "operation": "brightness", "params": {"gamma": 0.5}, "prob": 0.8},
            {"agent": "photometric", "operation": "brightness", "params": {"gamma": 0.6}, "prob": 0.7},
            {"agent": "photometric", "operation": "contrast", "params": {"clip_limit": 2.8}, "prob": 0.5},
        ],
        "blur": [
            {"agent": "photometric", "operation": "blur", "params": {"kernel": 7}, "prob": 0.6},
            {"agent": "generative", "operation": "elastic", "params": {"alpha": 25}, "prob": 0.3},
        ],
        "low_contrast": [
            {"agent": "photometric", "operation": "contrast", "params": {"clip_limit": 2.0}, "prob": 0.7},
            {"agent": "photometric", "operation": "clahe", "params": {"clip_limit": 3.0}, "prob": 0.5},
        ],
        "rotation": [
            {"agent": "geometric", "operation": "rotate", "params": {"angle": 20}, "prob": 0.8},
            {"agent": "geometric", "operation": "affine", "params": {"shear": 15}, "prob": 0.4},
        ],
        "scale": [
            {"agent": "geometric", "operation": "scale", "params": {"factor": 1.4}, "prob": 0.7},
            {"agent": "geometric", "operation": "crop", "params": {"ratio": 0.75}, "prob": 0.6},
        ],
        "color_cast": [
            {"agent": "photometric", "operation": "color_jitter", "params": {"hue": 10, "saturation": 1.4}, "prob": 0.7},
            {"agent": "photometric", "operation": "hsv_shift", "params": {"h_shift": 5}, "prob": 0.5},
        ],
        "occlusion": [
            {"agent": "generative", "operation": "synthetic_occlusion", "params": {"count": 2}, "prob": 0.6},
            {"agent": "generative", "operation": "inpaint", "params": {"mask_ratio": 0.15}, "prob": 0.4},
        ],
        "noise": [
            {"agent": "photometric", "operation": "noise", "params": {"sigma": 20}, "prob": 0.6},
            {"agent": "photometric", "operation": "denoise", "params": {}, "prob": 0.3},
        ],
        "default": [
            {"agent": "geometric", "operation": "flip", "params": {"direction": "horizontal"}, "prob": 0.5},
            {"agent": "photometric", "operation": "brightness", "params": {"gamma": 0.8}, "prob": 0.3},
        ]
    }

    def __init__(self):
        self.analyzer = ModelWeaknessAnalyzer()
        self.strategy_history: List[Dict[str, Any]] = []

    def plan_for_weaknesses(self, weaknesses: List[Dict[str, Any]], augmentations_per_weakness: int = 2, max_augmentations: int = 6) -> List[Dict[str, Any]]:
        """
        Select the right agents to target specific gaps.
        Returns ordered augmentation plan.
        """
        plan: List[Dict[str, Any]] = []
        for w in weaknesses[:3]:  # focus on top 3 weaknesses
            w_type = w.get("type", "default").lower()
            # fuzzy match
            key = "default"
            for k in self.STRATEGY_LIBRARY:
                if k in w_type or w_type in k:
                    key = k
                    break
            candidates = self.STRATEGY_LIBRARY.get(key, self.STRATEGY_LIBRARY["default"])
            # select based on prob and severity
            for cand in candidates[:augmentations_per_weakness]:
                if len(plan) >= max_augmentations:
                    break
                # copy and tag with weakness
                step = {**cand, "target_weakness": w_type, "severity": w.get("severity")}
                plan.append({"agent": step["agent"], "operation": step["operation"], "params": step["params"]})

        # Ensure diversity: if plan still small, add default
        if not plan:
            plan = [{"agent": "geometric", "operation": "flip", "params": {"direction": "horizontal"}}]

        self.strategy_history.append({"weaknesses": weaknesses, "plan": plan})
        return plan

    def plan_from_model_report(self, report: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        weaknesses = self.analyzer.analyze_from_report(report)
        return self.plan_for_weaknesses(weaknesses, **kwargs)

    def plan_from_dataset(self, image_folder: str, performance_json: str = None, **kwargs) -> List[Dict[str, Any]]:
        weaknesses = self.analyzer.analyze_from_folder(image_folder, performance_json)
        return self.plan_for_weaknesses(weaknesses, **kwargs)

    def plan_from_hint(self, hint: str) -> List[Dict[str, Any]]:
        """Simple hint-based planning e.g., 'low_light'."""
        weaknesses = [{"type": hint, "severity": 0.9}]
        return self.plan_for_weaknesses(weaknesses)

    def adaptive_plan(self, prior_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Close the loop - refine strategy based on quality gate feedback.
        If prior augmentations were rejected (low quality), reduce prob of that op.
        """
        # Example: if last plan had frequent rejections for blur, switch to milder blur
        # For now, heuristic: if many failures, fallback to safer flips/brighter
        if not prior_results:
            return self.plan_from_hint("default")
        failed_ops = [r for r in prior_results if not r.get("quality", {}).get("final_pass", True)]
        if len(failed_ops) > len(prior_results)//2:
            # Too many failures -> conservative plan
            return [
                {"agent": "geometric", "operation": "flip", "params": {"direction": "horizontal"}},
                {"agent": "photometric", "operation": "contrast", "params": {"clip_limit": 2.0}},
            ]
        return self.plan_from_hint("default")
