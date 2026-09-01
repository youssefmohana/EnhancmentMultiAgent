"""Metrics for built-in benchmarking & reporting."""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from typing import Dict, Any, List


def compute_metrics(original_path: str, augmented_path: str) -> Dict[str, Any]:
    """Compute PSNR, SSIM, MSE, MAE, edge preservation for augmentation evaluation."""
    orig = cv2.imread(original_path)
    rest = cv2.imread(augmented_path)
    if orig is None or rest is None:
        return {"error": "Could not load images"}
    if orig.shape != rest.shape:
        rest = cv2.resize(rest, (orig.shape[1], orig.shape[0]))
    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    rest_gray = cv2.cvtColor(rest, cv2.COLOR_BGR2GRAY)
    psnr_val = float(psnr(orig, rest, data_range=255))
    ssim_val = float(ssim(orig_gray, rest_gray, data_range=255))
    mse = float(np.mean((orig.astype(np.float32) - rest.astype(np.float32))**2))
    mae = float(np.mean(np.abs(orig.astype(np.float32) - rest.astype(np.float32))))
    # edge diff
    orig_edges = cv2.Canny(orig_gray, 100, 200)
    rest_edges = cv2.Canny(rest_gray, 100, 200)
    edge_diff = float(np.mean(np.abs(orig_edges.astype(np.float32) - rest_edges.astype(np.float32))))
    # diversity metric: histogram distance
    hist_orig = cv2.calcHist([orig], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    hist_aug = cv2.calcHist([rest], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    hist_orig = cv2.normalize(hist_orig, hist_orig).flatten()
    hist_aug = cv2.normalize(hist_aug, hist_aug).flatten()
    hist_corr = float(cv2.compareHist(hist_orig, hist_aug, cv2.HISTCMP_CORREL))

    return {
        "psnr": round(psnr_val,2),
        "ssim": round(ssim_val,4),
        "mse": round(mse,2),
        "mae": round(mae,2),
        "edge_diff": round(edge_diff,2),
        "hist_correlation": round(hist_corr,4),
        "diversity": round(1.0 - hist_corr,4),  # higher = more diverse augmentation
    }


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate list of per-image metric dicts."""
    valid = [r["metrics"] for r in results if "error" not in r.get("metrics", {})]
    if not valid:
        return {}
    keys = ["psnr", "ssim", "mse", "mae", "edge_diff", "diversity"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in valid if k in m]
        if vals:
            agg[k] = {"avg": round(sum(vals)/len(vals),3), "min": round(min(vals),3), "max": round(max(vals),3)}
    agg["count"] = len(valid)
    agg["total"] = len(results)
    return agg
