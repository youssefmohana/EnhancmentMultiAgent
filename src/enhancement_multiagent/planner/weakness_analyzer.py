"""Model weakness analyzer - analyse where model fails (e.g., poor low-light)."""
import json
import os
from typing import Dict, Any, List
from collections import Counter
import cv2
import numpy as np


class ModelWeaknessAnalyzer:
    """
    Analyzes model weaknesses from:
    - performance logs / per-class accuracy
    - confusion matrix
    - dataset distribution stats
    - image quality stats
    Target: close the loop - find gaps to select right agents.
    """

    def analyze_from_report(self, performance_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        performance_report example:
        {
          "per_class_accuracy": {"low_light": 0.52, "normal": 0.91, "blurry": 0.61},
          "per_condition": {"small_objects": 0.58, ...},
          "confusion_matrix": [...]
        }
        Returns list of weaknesses sorted by severity.
        """
        weaknesses = []
        per_class = performance_report.get("per_class_accuracy") or performance_report.get("per_condition") or {}
        for condition, acc in per_class.items():
            if acc < 0.75:  # threshold
                severity = 1.0 - acc
                weaknesses.append({"type": condition, "accuracy": acc, "severity": round(severity,3), "suggested_agents": self._map_condition_to_agents(condition)})

        # Also check overall failure modes from explicit list
        explicit = performance_report.get("weaknesses") or performance_report.get("failure_modes")
        if explicit:
            for w in explicit:
                if isinstance(w, str):
                    weaknesses.append({"type": w, "severity": 0.8, "suggested_agents": self._map_condition_to_agents(w)})
                elif isinstance(w, dict):
                    w.setdefault("suggested_agents", self._map_condition_to_agents(w.get("type","unknown")))
                    weaknesses.append(w)

        # Sort by severity
        weaknesses = sorted(weaknesses, key=lambda x: x.get("severity",0), reverse=True)
        return weaknesses

    def analyze_dataset_distribution(self, image_paths: List[str], sample_size: int = 100) -> List[Dict[str, Any]]:
        """
        Analyze dataset images to find underrepresented conditions.
        Cheap heuristic: brightness, blur, contrast distribution.
        """
        weaknesses = []
        if not image_paths:
            return weaknesses
        # sample
        paths = image_paths[:sample_size]
        brightness_vals = []
        blur_vals = []
        contrast_vals = []
        for p in paths:
            try:
                img = cv2.imread(p)
                if img is None: continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness_vals.append(float(gray.mean()))
                blur_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                contrast_vals.append(float(gray.std()))
            except Exception:
                continue
        if not brightness_vals:
            return weaknesses
        # Find underrepresented extremes (few low-light etc.)
        low_light_ratio = sum(1 for b in brightness_vals if b < 70) / len(brightness_vals)
        high_blur_ratio = sum(1 for v in blur_vals if v < 80) / len(blur_vals)
        low_contrast_ratio = sum(1 for c in contrast_vals if c < 35) / len(contrast_vals)

        if low_light_ratio < 0.15:
            weaknesses.append({"type": "low_light", "severity": 0.7, "ratio": low_light_ratio, "suggested_agents": ["photometric"], "reason": f"Only {low_light_ratio*100:.1f}% low-light images, model may fail in dark"})
        if high_blur_ratio < 0.2:
            weaknesses.append({"type": "blur_robustness", "severity": 0.6, "ratio": high_blur_ratio, "suggested_agents": ["photometric", "generative"], "reason": "Few blurry samples, missing blur invariance"})
        if low_contrast_ratio < 0.15:
            weaknesses.append({"type": "low_contrast", "severity": 0.5, "ratio": low_contrast_ratio, "suggested_agents": ["photometric"], "reason": "Low contrast underrepresented"})

        return sorted(weaknesses, key=lambda x: x["severity"], reverse=True)

    def analyze_from_folder(self, folder: str, performance_json: str = None) -> List[Dict[str, Any]]:
        dataset_weaknesses = []
        if os.path.isdir(folder):
            exts = (".png",".jpg",".jpeg",".bmp",".webp")
            files = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(exts)]
            dataset_weaknesses = self.analyze_dataset_distribution(files)
        model_weaknesses = []
        if performance_json and os.path.exists(performance_json):
            try:
                with open(performance_json) as f:
                    data = json.load(f)
                model_weaknesses = self.analyze_from_report(data)
            except Exception:
                pass
        # Merge
        return sorted(model_weaknesses + dataset_weaknesses, key=lambda x: x.get("severity",0), reverse=True)

    def _map_condition_to_agents(self, condition: str) -> List[str]:
        c = condition.lower()
        if "light" in c or "dark" in c or "bright" in c:
            return ["photometric"]
        if "blur" in c or "sharp" in c:
            return ["photometric", "generative"]
        if "contrast" in c:
            return ["photometric"]
        if "rotate" in c or "scale" in c or "crop" in c or "geometric" in c or "perspective" in c:
            return ["geometric"]
        if "occlu" in c or "small" in c:
            return ["generative", "geometric"]
        if "color" in c:
            return ["photometric"]
        if "noise" in c:
            return ["photometric"]
        return ["photometric", "geometric"]
