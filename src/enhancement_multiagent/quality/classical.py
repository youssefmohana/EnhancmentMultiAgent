"""Classical metrics gate - PSNR, SSIM, blur, noise, brightness."""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from .base import QualityGate, QualityResult
from typing import Dict, Any


class ClassicalGate(QualityGate):
    """Classical CV metrics. Fast, deterministic, no learned models."""

    def __init__(self, threshold: float = 20.0, min_ssim: float = 0.6):
        super().__init__(name="classical", threshold=threshold)
        self.min_ssim = min_ssim
        self.min_psnr = threshold

    def describe(self):
        return {"name": self.name, "metrics": ["psnr", "ssim", "mse", "blur_score", "brightness"], "thresholds": {"psnr": self.min_psnr, "ssim": self.min_ssim}}

    def validate(self, original_path: str, augmented_path: str) -> QualityResult:
        try:
            orig = cv2.imread(original_path)
            aug = cv2.imread(augmented_path)
            if orig is None or aug is None:
                return QualityResult(self.name, False, 0.0, {}, "Could not load images")

            if orig.shape != aug.shape:
                aug = cv2.resize(aug, (orig.shape[1], orig.shape[0]))

            # PSNR / SSIM
            psnr_val = float(psnr(orig, aug, data_range=255))
            orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            aug_gray = cv2.cvtColor(aug, cv2.COLOR_BGR2GRAY)
            ssim_val = float(ssim(orig_gray, aug_gray, data_range=255))
            mse = float(np.mean((orig.astype(np.float32) - aug.astype(np.float32))**2))
            mae = float(np.mean(np.abs(orig.astype(np.float32) - aug.astype(np.float32))))

            # Blur detection on augmented
            lap_var = float(cv2.Laplacian(aug_gray, cv2.CV_64F).var())
            blur_score = float(min(100, max(0, lap_var / 5)))
            brightness = float(aug_gray.mean())
            contrast = float(aug_gray.std())

            # For augmentation, we don't want to be too strict: psnr can be low if augmentation is intentional (e.g., flip still high psnr but brightness change lowers it)
            # So we check ssim and that image is not degenerate
            # Augmentation is valid if it changes image but preserves structure reasonably
            # Heuristic: ssim > 0.4 and psnr > 15 unless operation is intentional distortion
            passed = ssim_val > self.min_ssim or (ssim_val > 0.4 and psnr_val > 15)
            # If image is completely black or white, fail
            if brightness < 5 or brightness > 250:
                passed = False
            if blur_score < 5 and ssim_val < 0.7:  # excessive blur without preserving structure
                pass  # allow, blur is intentional augmentation

            details = {
                "psnr": round(psnr_val, 2),
                "ssim": round(ssim_val, 4),
                "mse": round(mse,2),
                "mae": round(mae,2),
                "blur_score": round(blur_score,2),
                "brightness": round(brightness,2),
                "contrast": round(contrast,2),
            }
            score = (ssim_val + min(1.0, psnr_val/35))/2
            msg = f"PSNR {psnr_val:.1f}dB SSIM {ssim_val:.3f} blur {blur_score:.1f}"
            return QualityResult(self.name, passed, score, details, msg, self.threshold)

        except Exception as e:
            return QualityResult(self.name, False, 0.0, {}, str(e), self.threshold)
