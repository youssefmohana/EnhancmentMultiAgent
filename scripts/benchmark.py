#!/usr/bin/env python3
"""
Benchmark Evaluation Script
Runs the multi-agent restoration pipeline on all benchmark images
and computes PSNR, SSIM, and quality metrics.
"""
import asyncio
import json
import os
import time
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import our restoration system (supports both new and legacy paths)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from enhancement_multiagent.pipelines.restoration import restore_image
except ImportError:
    from image_restoration import restore_image  # legacy shim fallback

BENCHMARK_DIR = "benchmark"
ORIGINAL_DIR = os.path.join(BENCHMARK_DIR, "original")
DEGRADED_DIR = os.path.join(BENCHMARK_DIR, "degraded")
RESTORED_DIR = os.path.join(BENCHMARK_DIR, "restored")
REPORTS_DIR = "reports"

os.makedirs(RESTORED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def compute_metrics(original_path: str, restored_path: str) -> dict:
    """Compute PSNR, SSIM, and other quality metrics."""
    orig = cv2.imread(original_path)
    rest = cv2.imread(restored_path)

    if orig is None or rest is None:
        return {"error": "Could not load images"}

    # Ensure same size
    if orig.shape != rest.shape:
        rest = cv2.resize(rest, (orig.shape[1], orig.shape[0]))

    # Convert to grayscale for some metrics
    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    rest_gray = cv2.cvtColor(rest, cv2.COLOR_BGR2GRAY)

    # PSNR
    psnr_val = psnr(orig, rest)

    # SSIM
    ssim_val = ssim(orig_gray, rest_gray, data_range=255)

    # Mean Squared Error
    mse = np.mean((orig.astype(np.float32) - rest.astype(np.float32)) ** 2)

    # Mean Absolute Error
    mae = np.mean(np.abs(orig.astype(np.float32) - rest.astype(np.float32)))

    # LPIPS-like simple metric (perceptual distance via edge comparison)
    orig_edges = cv2.Canny(orig_gray, 100, 200)
    rest_edges = cv2.Canny(rest_gray, 100, 200)
    edge_diff = np.mean(np.abs(orig_edges.astype(np.float32) - rest_edges.astype(np.float32)))

    return {
        "psnr": round(float(psnr_val), 2),
        "ssim": round(float(ssim_val), 4),
        "mse": round(float(mse), 2),
        "mae": round(float(mae), 2),
        "edge_diff": round(float(edge_diff), 2),
    }


def generate_report(results: list, output_path: str):
    """Generate a markdown report with tables and charts."""

    # Compute averages
    psnr_vals = [r["metrics"]["psnr"] for r in results if "error" not in r["metrics"]]
    ssim_vals = [r["metrics"]["ssim"] for r in results if "error" not in r["metrics"]]
    mse_vals = [r["metrics"]["mse"] for r in results if "error" not in r["metrics"]]

    avg_psnr = round(sum(psnr_vals) / len(psnr_vals), 2) if psnr_vals else 0
    avg_ssim = round(sum(ssim_vals) / len(ssim_vals), 4) if ssim_vals else 0
    avg_mse = round(sum(mse_vals) / len(mse_vals), 2) if mse_vals else 0

    # Best and worst
    best_psnr = max(psnr_vals) if psnr_vals else 0
    worst_psnr = min(psnr_vals) if psnr_vals else 0
    best_ssim = max(ssim_vals) if ssim_vals else 0
    worst_ssim = min(ssim_vals) if ssim_vals else 0

    report = f"""# 🖼️ Image Restoration Benchmark Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Total Images:** {len(results)}  
**System:** Ollama + FastMCP + OpenCV Multi-Agent Pipeline

---

## 📊 Summary Statistics

| Metric | Average | Best | Worst |
|--------|---------|------|-------|
| **PSNR (dB)** | {avg_psnr} | {best_psnr} | {worst_psnr} |
| **SSIM** | {avg_ssim} | {best_ssim} | {worst_ssim} |
| **MSE** | {avg_mse} | — | — |

### Interpretation
- **PSNR > 30 dB**: Excellent restoration
- **PSNR 25-30 dB**: Good restoration
- **PSNR 20-25 dB**: Fair restoration
- **PSNR < 20 dB**: Poor restoration

- **SSIM > 0.90**: Excellent structural similarity
- **SSIM 0.80-0.90**: Good similarity
- **SSIM 0.70-0.80**: Fair similarity
- **SSIM < 0.70**: Poor similarity

---

## 📋 Per-Image Results

| # | Image | PSNR ↑ | SSIM ↑ | MSE ↓ | Time (s) | Status |
|---|-------|--------|--------|-------|----------|--------|
"""

    for i, r in enumerate(results, 1):
        if "error" in r["metrics"]:
            report += f"| {i} | `{r['image']}` | — | — | — | {r['time']:.1f} | ❌ Failed |\n"
        else:
            report += f"| {i} | `{r['image']}` | {r['metrics']['psnr']} | {r['metrics']['ssim']} | {r['metrics']['mse']} | {r['time']:.1f} | ✅ OK |\n"

    report += f"""
---

## 🔍 Detailed Findings

"""

    # Find best and worst performers
    valid_results = [r for r in results if "error" not in r["metrics"]]
    if valid_results:
        best = max(valid_results, key=lambda x: x["metrics"]["psnr"])
        worst = min(valid_results, key=lambda x: x["metrics"]["psnr"])

        report += f"""### 🏆 Best Performing Image
- **Image:** `{best['image']}`
- **PSNR:** {best['metrics']['psnr']} dB
- **SSIM:** {best['metrics']['ssim']}
- **Why it worked well:** The degradation was mild enough for the pipeline to fully recover details.

### ⚠️ Worst Performing Image
- **Image:** `{worst['image']}`
- **PSNR:** {worst['metrics']['psnr']} dB
- **SSIM:** {worst['metrics']['ssim']}
- **Why it struggled:** Severe degradation (heavy blur + noise + downscaling) exceeded the recovery capacity of the OpenCV-based tools.

"""

    report += f"""---

## 🛠️ System Configuration

| Component | Version/Details |
|-----------|----------------|
| LLM Engine | Ollama (llama3.2) |
| Tool Protocol | MCP (FastMCP) |
| Image Processing | OpenCV 4.x |
| Agents | 4 (Diagnosis, Planning, Restoration, Evaluation) |

---

## 📈 Recommendations

1. **For severe blur:** Consider adding deconvolution-based deblurring (Wiener filter, Richardson-Lucy)
2. **For heavy noise:** Implement BM3D or deep learning denoising (DnCNN)
3. **For super-resolution:** Replace Lanczos with ESRGAN or Real-ESRGAN for 4x+ upscaling
4. **For color correction:** Add histogram matching against reference images

---

*Generated by Multi-Agent Image Restoration Benchmark Suite*
"""

    with open(output_path, "w") as f:
        f.write(report)

    return report


def generate_csv(results: list, output_path: str):
    """Export results to CSV for further analysis."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "psnr", "ssim", "mse", "mae", "edge_diff", "time_seconds", "status"])
        for r in results:
            if "error" in r["metrics"]:
                writer.writerow([r["image"], "", "", "", "", "", f"{r['time']:.2f}", "failed"])
            else:
                writer.writerow([
                    r["image"],
                    r["metrics"]["psnr"],
                    r["metrics"]["ssim"],
                    r["metrics"]["mse"],
                    r["metrics"]["mae"],
                    r["metrics"]["edge_diff"],
                    f"{r['time']:.2f}",
                    "success"
                ])


def generate_visualization(results: list, output_path: str):
    """Generate a bar chart of PSNR and SSIM per image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = [r for r in results if "error" not in r["metrics"]]
        if not valid:
            return

        names = [r["image"][:15] for r in valid]
        psnr_vals = [r["metrics"]["psnr"] for r in valid]
        ssim_vals = [r["metrics"]["ssim"] for r in valid]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # PSNR chart
        colors = ["green" if p > 30 else "orange" if p > 25 else "red" for p in psnr_vals]
        axes[0].bar(range(len(names)), psnr_vals, color=colors, edgecolor="black", linewidth=0.5)
        axes[0].axhline(y=30, color="green", linestyle="--", alpha=0.5, label="Excellent (30 dB)")
        axes[0].axhline(y=25, color="orange", linestyle="--", alpha=0.5, label="Good (25 dB)")
        axes[0].set_xticks(range(len(names)))
        axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("PSNR (dB)")
        axes[0].set_title("Peak Signal-to-Noise Ratio per Image")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)

        # SSIM chart
        colors2 = ["green" if s > 0.9 else "orange" if s > 0.8 else "red" for s in ssim_vals]
        axes[1].bar(range(len(names)), ssim_vals, color=colors2, edgecolor="black", linewidth=0.5)
        axes[1].axhline(y=0.9, color="green", linestyle="--", alpha=0.5, label="Excellent (0.90)")
        axes[1].axhline(y=0.8, color="orange", linestyle="--", alpha=0.5, label="Good (0.80)")
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("SSIM")
        axes[1].set_title("Structural Similarity Index per Image")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   📊 Visualization saved: {output_path}")
    except Exception as e:
        print(f"   ⚠️  Could not generate visualization: {e}")


async def run_benchmark():
    print("=" * 60)
    print("🏃 BENCHMARK EVALUATION")
    print("=" * 60)

    # Get all degraded images
    degraded_files = sorted([f for f in os.listdir(DEGRADED_DIR) 
                            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])

    if not degraded_files:
        print("❌ No degraded images found in benchmark/degraded/")
        print("   Run: python download_benchmark.py")
        return

    print(f"📁 Found {len(degraded_files)} images to restore")
    print("")

    results = []

    for i, filename in enumerate(degraded_files, 1):
        degraded_path = os.path.join(DEGRADED_DIR, filename)
        original_path = os.path.join(ORIGINAL_DIR, filename)

        print(f"[{i}/{len(degraded_files)}] Processing: {filename}")

        start_time = time.time()

        try:
            # Run restoration
            restored_path = await restore_image(degraded_path)

            # Move to benchmark/restored with proper name
            final_restored = os.path.join(RESTORED_DIR, filename)
            if os.path.exists(restored_path):
                import shutil
                shutil.copy(restored_path, final_restored)

            elapsed = time.time() - start_time

            # Compute metrics if original exists
            if os.path.exists(original_path):
                metrics = compute_metrics(original_path, final_restored)
                print(f"      PSNR: {metrics['psnr']} dB | SSIM: {metrics['ssim']} | Time: {elapsed:.1f}s")
            else:
                metrics = {"error": "Original image not found for comparison"}
                print(f"      ⚠️  No original for comparison | Time: {elapsed:.1f}s")

            results.append({
                "image": filename,
                "metrics": metrics,
                "time": elapsed,
                "restored_path": final_restored
            })

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"      ❌ Failed: {e}")
            results.append({
                "image": filename,
                "metrics": {"error": str(e)},
                "time": elapsed,
                "restored_path": ""
            })

        print("")

    # Generate reports
    print("=" * 60)
    print("📊 Generating reports...")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Markdown report
    md_path = os.path.join(REPORTS_DIR, f"benchmark_report_{timestamp}.md")
    generate_report(results, md_path)
    print(f"   ✓ Markdown report: {md_path}")

    # Also save as latest
    latest_md = os.path.join(REPORTS_DIR, "benchmark_report.md")
    generate_report(results, latest_md)

    # CSV export
    csv_path = os.path.join(REPORTS_DIR, f"benchmark_results_{timestamp}.csv")
    generate_csv(results, csv_path)
    print(f"   ✓ CSV export: {csv_path}")

    # Visualization
    viz_path = os.path.join(REPORTS_DIR, f"benchmark_chart_{timestamp}.png")
    generate_visualization(results, viz_path)

    # Print summary
    valid_results = [r for r in results if "error" not in r["metrics"]]
    if valid_results:
        avg_psnr = sum(r["metrics"]["psnr"] for r in valid_results) / len(valid_results)
        avg_ssim = sum(r["metrics"]["ssim"] for r in valid_results) / len(valid_results)
        total_time = sum(r["time"] for r in results)

        print(f"""
{'=' * 60}
📈 BENCHMARK SUMMARY
{'=' * 60}
   Images processed: {len(results)}
   Successful: {len(valid_results)}
   Failed: {len(results) - len(valid_results)}

   Average PSNR: {avg_psnr:.2f} dB
   Average SSIM: {avg_ssim:.4f}
   Total time: {total_time:.1f}s
   Avg per image: {total_time / len(results):.1f}s
{'=' * 60}""")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
