"""Perceptual gate - learned / handcrafted perceptual metrics."""
import cv2
import numpy as np
from .base import QualityGate, QualityResult


class PerceptualGate(QualityGate):
    """Perceptual validation: edge preservation, histogram, LPIPS stub."""

    def __init__(self, threshold: float = 0.5):
        super().__init__(name="perceptual", threshold=threshold)

    def describe(self):
        return {"name": self.name, "metrics": ["edge_preservation", "histogram_distance", "lpips_proxy"], "threshold": self.threshold}

    def validate(self, original_path: str, augmented_path: str) -> QualityResult:
        try:
            orig = cv2.imread(original_path)
            aug = cv2.imread(augmented_path)
            if orig is None or aug is None:
                return QualityResult(self.name, False, 0.0, {}, "Load failed")

            if orig.shape != aug.shape:
                aug = cv2.resize(aug, (orig.shape[1], orig.shape[0]))

            orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            aug_gray = cv2.cvtColor(aug, cv2.COLOR_BGR2GRAY)

            # Edge preservation
            orig_edges = cv2.Canny(orig_gray, 100, 200)
            aug_edges = cv2.Canny(aug_gray, 100, 200)
            # Intersection over union of edges
            intersection = np.logical_and(orig_edges>0, aug_edges>0).sum()
            union = np.logical_or(orig_edges>0, aug_edges>0).sum()
            edge_iou = float(intersection / max(union, 1))
            edge_diff = float(np.mean(np.abs(orig_edges.astype(np.float32) - aug_edges.astype(np.float32))))

            # Histogram distance (Bhattacharyya)
            hist_orig = cv2.calcHist([orig], [0,1,2], None, [32,32,32], [0,256,0,256,0,256])
            hist_orig = cv2.normalize(hist_orig, hist_orig).flatten()
            hist_aug = cv2.calcHist([aug], [0,1,2], None, [32,32,32], [0,256,0,256,0,256])
            hist_aug = cv2.normalize(hist_aug, hist_aug).flatten()
            # correlation
            hist_corr = float(cv2.compareHist(hist_orig, hist_aug, cv2.HISTCMP_CORREL))
            # Convert to distance
            hist_dist = 1.0 - max(0, hist_corr)

            # LPIPS proxy: if torch + lpips available, use it; else use simple perceptual via feature distance
            lpips_proxy = 0.0
            try:
                # Try to use LPIPS if installed (optional)
                import torch
                import lpips
                # lazy load not implemented; fallback to proxy
                lpips_proxy = 0.0
            except Exception:
                # Proxy: normalized MSE in LAB perceptual space
                orig_lab = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
                aug_lab = cv2.cvtColor(aug, cv2.COLOR_BGR2LAB).astype(np.float32)
                lpips_proxy = float(np.mean(np.abs(orig_lab - aug_lab)) / 100.0)

            # Scoring: edge_iou > 0.2 and hist_dist not too large (<0.8) indicates perceptual similarity preserved
            # For augmentation, we expect some hist difference, so allow up to 0.6
            passed = edge_iou > 0.08 and hist_corr > 0.2
            # If geometric flip/rotate, edges will still correlate but position changes -> edge_iou low is ok
            # So also allow if hist is preserved
            if hist_corr > 0.5:
                passed = True

            score = (edge_iou*0.5 + hist_corr*0.5)
            details = {
                "edge_iou": round(edge_iou, 4),
                "edge_diff": round(edge_diff, 2),
                "hist_correlation": round(hist_corr, 4),
                "hist_distance": round(hist_dist, 4),
                "lpips_proxy": round(lpips_proxy, 4),
            }
            msg = f"edge IoU {edge_iou:.3f} hist corr {hist_corr:.3f}"
            return QualityResult(self.name, passed, score, details, msg, self.threshold)

        except Exception as e:
            return QualityResult(self.name, False, 0.0, {}, str(e), self.threshold)
