#!/bin/bash
# =================================================================
# Multi-Agent Image Restoration System - Runner Script
# Ollama + FastMCP + OpenCV
# =================================================================

set -e  # Exit on error

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

echo "========================================"
echo "🖼️  Image Restoration Multi-Agent System"
echo "========================================"

# ── Check dependencies ──
echo ""
echo "🔍 Checking dependencies..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi
echo "   ✓ Python found"

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is required. Install from https://ollama.com"
    exit 1
fi
echo "   ✓ Ollama found"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   ⚠️  Starting Ollama server..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 3
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "❌ Failed to start Ollama. Please start it manually: ollama serve"
        exit 1
    fi
    echo "   ✓ Ollama started (PID: $OLLAMA_PID)"
else
    echo "   ✓ Ollama is running"
fi

# Check required models
for model in llama3.2; do
    if ! ollama list | grep -q "$model"; then
        echo "   📥 Pulling model: $model..."
        ollama pull "$model"
    else
        echo "   ✓ Model '$model' available"
    fi
done

# Check Python packages
python3 -c "import cv2, numpy, ollama, mcp" 2>/dev/null || {
    echo ""
    echo "📦 Installing Python dependencies..."
    pip install -q opencv-python-headless numpy ollama mcp pillow matplotlib
}
echo "   ✓ Python packages OK"

# ── Create directories ──
mkdir -p demo_images restored reports benchmark/degraded benchmark/original benchmark/restored

# ── Parse arguments ──
MODE="${1:-demo}"
INPUT="${2:-}"

echo ""
echo "========================================"

 case "$MODE" in
    demo)
        echo "🎯 MODE: Demo (synthetic degraded image)"
        echo "========================================"
        python3 image_restoration.py
        ;;

    restore)
        if [ -z "$INPUT" ]; then
            echo "❌ Usage: ./run.sh restore <image_path>"
            echo "   Example: ./run.sh restore my_photo.jpg"
            exit 1
        fi
        if [ ! -f "$INPUT" ]; then
            echo "❌ File not found: $INPUT"
            exit 1
        fi
        echo "🎯 MODE: Single Image Restoration"
        echo "   Input: $INPUT"
        echo "========================================"
        python3 image_restoration.py "$INPUT"
        ;;

    benchmark)
        echo "🎯 MODE: Benchmark Evaluation"
        echo "========================================"

        # Download benchmark if not exists
        if [ ! -d "benchmark/original" ] || [ -z "$(ls -A benchmark/original 2>/dev/null)" ]; then
            echo ""
            echo "📥 Downloading benchmark dataset..."
            python3 download_benchmark.py
        fi

        echo ""
        echo "🏃 Running benchmark restoration..."
        python3 benchmark.py

        echo ""
        echo "📊 Report saved to: reports/benchmark_report.md"
        ;;

    batch)
        if [ -z "$INPUT" ]; then
            echo "❌ Usage: ./run.sh batch <folder_path>"
            echo "   Example: ./run.sh batch ./my_photos/"
            exit 1
        fi
        if [ ! -d "$INPUT" ]; then
            echo "❌ Directory not found: $INPUT"
            exit 1
        fi
        echo "🎯 MODE: Batch Processing"
        echo "   Folder: $INPUT"
        echo "========================================"
        python3 batch_restore.py "$INPUT"
        ;;

    clean)
        echo "🧹 Cleaning generated files..."
        rm -rf restored/* benchmark/restored/* reports/*
        echo "   ✓ Cleaned restored/, benchmark/restored/, reports/"
        ;;

    help|--help|-h)
        echo "Usage: ./run.sh [MODE] [INPUT]"
        echo ""
        echo "Modes:"
        echo "  demo       Run with synthetic degraded image (default)"
        echo "  restore    Restore a single image: ./run.sh restore photo.jpg"
        echo "  benchmark  Download dataset and run full benchmark evaluation"
        echo "  batch      Process all images in a folder: ./run.sh batch ./photos/"
        echo "  clean      Remove all generated outputs"
        echo "  help       Show this help"
        echo ""
        echo "Examples:"
        echo "  ./run.sh                          # Run demo"
        echo "  ./run.sh restore my_photo.jpg     # Restore one image"
        echo "  ./run.sh benchmark                # Full benchmark"
        echo "  ./run.sh batch ./vacation_photos/ # Batch restore folder"
        ;;

    *)
        echo "❌ Unknown mode: $MODE"
        echo "   Run './run.sh help' for usage."
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "✅ Done!"
echo "========================================"
