# 🖼️ Multi-Agent Image Restoration System

**Ollama + FastMCP + OpenCV**

A complete multi-agent image restoration pipeline that automatically diagnoses degraded photos and applies the right fixes using 4 specialized AI agents.

---

## 🏗️ Architecture

```
User Image
    ↓
┌─────────────────────────────────────┐
│  MCP Image Tool Server (OpenCV)    │
│  • analyze_image_quality            │
│  • denoise_image                    │
│  • deblur_image                     │
│  • upscale_image                    │
│  • enhance_contrast                 │
│  • color_correct                    │
│  • sharpen_image                    │
│  • adjust_brightness                │
└─────────────────────────────────────┘
    ↑
    │  (MCP protocol - stdio transport)
    ↓
┌─────────────────────────────────────┐
│  Multi-Agent Orchestrator           │
│                                     │
│  1. Diagnosis Agent  → analyzes     │
│  2. Planning Agent   → plans steps  │
│  3. Restoration Agent → executes  │
│  4. Evaluation Agent → verifies     │
│                                     │
│  Powered by Ollama (llama3.2)       │
└─────────────────────────────────────┘
```

---

## 📦 Installation

### 1. Install Ollama
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com
```

### 2. Pull the model
```bash
ollama pull llama3.2
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Run with a single command
```bash
./run.sh
```

### Or manually
```bash
# Demo mode (creates synthetic degraded image)
python image_restoration.py

# Restore your own image
python image_restoration.py my_photo.jpg

# Batch process a folder
python batch_restore.py ./my_photos/

# Full benchmark evaluation
./run.sh benchmark
```

---

## 📁 Project Structure

```
image_restoration_app/
├── run.sh                      # Main runner script
├── requirements.txt            # Python dependencies
│
├── mcp_image_server.py         # MCP Tool Server (OpenCV tools)
├── image_restoration.py        # Multi-Agent Orchestrator
│
├── download_benchmark.py       # Download/generate benchmark dataset
├── benchmark.py                # Run benchmark + compute PSNR/SSIM
├── batch_restore.py            # Batch process folders
│
├── demo_images/                # Demo input images
├── restored/                   # Restored output images
├── benchmark/
│   ├── original/               # Ground truth images
│   ├── degraded/               # Synthetic degraded images
│   └── restored/               # Benchmark restoration results
└── reports/
    ├── benchmark_report.md     # Markdown report
    ├── benchmark_results.csv   # CSV export
    └── benchmark_chart.png     # Visualization chart
```

---

## 🎯 Usage Modes

### 1. Demo Mode
```bash
./run.sh demo
```
Creates a synthetic degraded image and restores it automatically.

### 2. Single Image
```bash
./run.sh restore my_photo.jpg
```
Restores any image you provide.

### 3. Batch Processing
```bash
./run.sh batch ./vacation_photos/
```
Processes all images in a folder.

### 4. Benchmark Evaluation
```bash
./run.sh benchmark
```
Downloads a benchmark dataset, runs restoration on all images, and generates:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- Per-image quality charts
- Full markdown report

### 5. Clean Outputs
```bash
./run.sh clean
```
Removes all generated files.

---

## 🔬 Benchmark Dataset

The benchmark system:
1. **Attempts to download** real classic test images (Lena, Baboon, Barbara, etc.)
2. **Falls back** to generating 10 diverse synthetic images with controlled degradation:
   - Gaussian blur (3×3 to 11×11)
   - Gaussian noise (σ = 10-35)
   - Gamma darkening (γ = 0.4-0.8)
   - Resolution downscaling (optional)

Each degraded image has a known ground truth, enabling objective PSNR/SSIM evaluation.

---

## 📊 Sample Benchmark Report

| Metric | Average | Best | Worst |
|--------|---------|------|-------|
| **PSNR** | 28.4 dB | 34.2 dB | 22.1 dB |
| **SSIM** | 0.87 | 0.96 | 0.71 |
| **MSE** | 145.3 | — | — |

---

## 🛠️ MCP Tools Available

| Tool | Description |
|------|-------------|
| `analyze_image_quality` | Detects blur, noise, brightness, contrast, color cast, resolution |
| `denoise_image` | Non-Local Means denoising |
| `deblur_image` | Unsharp mask deblurring |
| `upscale_image` | Lanczos interpolation upscaling |
| `enhance_contrast` | CLAHE adaptive histogram equalization |
| `color_correct` | Gray-world white balance |
| `sharpen_image` | Convolution sharpening |
| `adjust_brightness` | Gamma correction |
| `save_image` | Save final result |

---

## 🔑 Why MCP Architecture?

| | Without MCP | With MCP |
|---|---|---|
| Image tools | Hardcoded in agent | **Separate reusable server** |
| New filter? | Edit agent code | **Add 1 tool to server** |
| Other apps? | Can't share | **Any app connects** |
| Agents | Fixed tool set | **Dynamically discover** tools |

---

## 🚀 Next Steps

1. **Add deep learning models**: Replace OpenCV with ESRGAN, DnCNN, or Real-ESRGAN
2. **Vision diagnosis**: Use `llama3.2-vision` for visual quality assessment
3. **Video support**: Process frame-by-frame with temporal consistency
4. **Web UI**: Add a Gradio or Streamlit interface
5. **Cloud deployment**: Package as Docker container

---

## 📝 License

MIT License — free for personal and commercial use.
