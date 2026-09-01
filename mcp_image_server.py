#!/usr/bin/env python3
"""
MCP Image Restoration Server
Exposes OpenCV image processing tools via the Model Context Protocol.
Any agent can call these tools — they don't process images themselves.
"""
from mcp.server.fastmcp import FastMCP
import cv2
import numpy as np
import os
import json

mcp = FastMCP("image_restoration_tools")


@mcp.tool()
def analyze_image_quality(image_path: str) -> str:
    """
    Analyze image quality metrics: blur, noise, brightness, contrast,
    color balance, dynamic range, resolution. Returns JSON report.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return json.dumps({"error": f"Could not load image: {image_path}"})

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Blur (Laplacian variance)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_score = float(min(100, max(0, laplacian_var / 5)))

        # Noise (MAD of Laplacian)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        noise_est = float(np.median(np.abs(lap - np.median(lap))) / 0.6745)
        noise_score = float(min(100, max(0, noise_est * 2)))

        brightness = float(gray.mean())
        contrast = float(gray.std())
        min_val, max_val = int(gray.min()), int(gray.max())
        dynamic_range = int(max_val - min_val)

        b, g, r = cv2.split(img)
        color_balance = {
            "red_deviation": round(float(r.mean() - gray.mean()), 2),
            "green_deviation": round(float(g.mean() - gray.mean()), 2),
            "blue_deviation": round(float(b.mean() - gray.mean()), 2),
        }

        resolution = {
            "width": int(w), "height": int(h),
            "megapixels": round(w * h / 1_000_000, 2)
        }

        # Issue detection
        issues = []
        if blur_score < 30: issues.append("significant_blur")
        elif blur_score < 50: issues.append("mild_blur")
        if noise_score > 40: issues.append("heavy_noise")
        elif noise_score > 20: issues.append("moderate_noise")
        if brightness < 60: issues.append("underexposed")
        elif brightness > 200: issues.append("overexposed")
        if contrast < 30: issues.append("low_contrast")
        if dynamic_range < 100: issues.append("low_dynamic_range")
        if max(abs(v) for v in color_balance.values()) > 20: issues.append("color_cast")
        if w < 800 or h < 600: issues.append("low_resolution")

        report = {
            "blur_score": round(blur_score, 2),
            "noise_score": round(noise_score, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "dynamic_range": dynamic_range,
            "color_balance": color_balance,
            "resolution": resolution,
            "detected_issues": issues,
            "overall_quality": "poor" if len(issues) >= 4 else "fair" if len(issues) >= 2 else "good"
        }
        return json.dumps(report, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def denoise_image(input_path: str, output_path: str) -> str:
    """Apply Non-Local Means denoising. Preserves edges while removing noise."""
    try:
        img = cv2.imread(input_path)
        if img is None: return f"Error: Could not load {input_path}"
        result = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        cv2.imwrite(output_path, result)
        return f"Denoised → {output_path}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def deblur_image(input_path: str, output_path: str) -> str:
    """Apply unsharp mask to reduce blur and enhance edges."""
    try:
        img = cv2.imread(input_path)
        if img is None: return f"Error: Could not load {input_path}"
        gaussian = cv2.GaussianBlur(img, (0, 0), 3)
        result = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
        cv2.imwrite(output_path, result)
        return f"Deblurred → {output_path}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def upscale_image(input_path: str, output_path: str, scale_factor: float = 2.0) -> str:
    """Upscale image using Lanczos interpolation (high quality)."""
    try:
        img = cv2.imread(input_path)
        if img is None: return f"Error: Could not load {input_path}"
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        result = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(output_path, result)
        return f"Upscaled {w}x{h} → {new_w}x{new_h} → {output_path}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def enhance_contrast(input_path: str, output_path: str) -> str:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    try:
        img = cv2.imread(input_path)
        if img is None: return f"Error: Could not load {input_path}"
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        cv2.imwrite(output_path, result)
        return f"Contrast enhanced → {output_path}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def color_correct(input_path: str, output_path: str) -> str:
    """Apply gray-world white balance to correct color casts."""
    try:
        img = cv2.imread(input_path)
        if img is None: return f"Error: Could not load {input_path}"
        b, g, r = cv2.split(img)
        b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
        k = (b_mean + g_mean + r_mean) / 3

        b = np.clip(b * (k / max(b_mean, 1)), 0, 255).astype(np.uint8)
        g = np.clip(g * (k / max(g_mean, 1)), 0, 255).astype(np.uint8)
        r = np.clip(r * (k / max(r_mean, 1)), 0, 255).astype(np.uint8)

        result = cv2.merge([b, g, r])
        cv2.imwrite(output_path, result)
        return f"Color corrected → {output_path}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def sharpen_image(input_path: str, output_path: str) -> str:
    """Apply sharpening kernel convolution."""
    try:
        img = cv2.imread(input_path)
        if img is None: return f"Error: Could not load {input_path}"
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        result = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(output_path, result)
        return f"Sharpened → {output_path}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def adjust_brightness(input_path: str, output_path: str, gamma: float = 1.5) -> str:
    """Apply gamma correction to adjust brightness."""
    try:
        img = cv2.imread(input_path)
        if img is None: return f"Error: Could not load {input_path}"
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        result = cv2.LUT(img, table)
        cv2.imwrite(output_path, result)
        return f"Brightness adjusted (γ={gamma}) → {output_path}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def save_image(input_path: str, output_path: str) -> str:
    """Copy/save image to final destination."""
    try:
        img = cv2.imread(input_path)
        if img is None: return f"Error: Could not load {input_path}"
        cv2.imwrite(output_path, img)
        return f"Saved final image → {output_path}"
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
