#!/bin/bash
# =================================================================
# 🧠 Enhancement MultiAgent - Runner (Senior Edition)
# Ollama + FastMCP + OpenCV + Vision LLM
# =================================================================

set -e
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

# Colors (fallback if not TTY)
GREEN="\033[0;32m"; CYAN="\033[0;36m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; NC="\033[0m"

echo "========================================"
echo -e "${CYAN}🧠 Enhancement MultiAgent${NC}  v0.2.0"
echo -e "${CYAN}🖼️  Smart Data Augmentation • Vision LLM${NC}"
echo "========================================"

# ── Check dependencies ──
echo ""
echo "🔍 Checking dependencies..."

if ! command -v python3 &> /dev/null; then
  if command -v python &> /dev/null; then alias python3=python; else echo -e "${RED}❌ Python 3 required${NC}"; exit 1; fi
fi
echo -e "${GREEN}   ✓ Python found${NC} $(python3 --version 2>&1 | head -n1)"

if ! command -v ollama &> /dev/null; then
  echo -e "${YELLOW}⚠️  Ollama not found — install from https://ollama.com (pipeline will use heuristic fallback)${NC}"
else
  echo -e "${GREEN}   ✓ Ollama found${NC}"
  if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}   ⚠️  Starting Ollama...${NC}"
    ollama serve & OLLAMA_PID=$!; sleep 3
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then echo -e "${GREEN}   ✓ Ollama started (PID $OLLAMA_PID)${NC}"; fi
  else echo -e "${GREEN}   ✓ Ollama running${NC}"; fi
  for model in llama3.2 llava; do
    if ollama list 2>/dev/null | grep -q "$model"; then echo -e "${GREEN}   ✓ Model '$model'${NC}"; else echo -e "${YELLOW}   ○ Model '$model' not found (optional)${NC}"; fi
  done
fi

python3 -c "import cv2, numpy, ollama, mcp" 2>/dev/null || {
  echo ""; echo "📦 Installing Python dependencies..."
  if command -v uv &> /dev/null; then uv sync 2>/dev/null || pip install -q -r requirements.txt; else pip install -q -r requirements.txt; fi
}
echo -e "${GREEN}   ✓ Python packages OK${NC}"

mkdir -p demo_images restored reports benchmark/degraded benchmark/original benchmark/restored

MODE="${1:-demo}"
INPUT="${2:-}"
EXTRA="${3:-}"

echo ""
echo "========================================"

case "$MODE" in
  demo)
    echo -e "${CYAN}🎯 MODE: Demo (restoration)${NC}"
    echo "========================================"
    python3 src/enhancement_multiagent/pipelines/restoration.py
    ;;
  restore)
    if [ -z "$INPUT" ]; then echo -e "${RED}❌ Usage: ./run.sh restore <image_path>${NC}"; exit 1; fi
    [ -f "$INPUT" ] || { echo -e "${RED}❌ File not found: $INPUT${NC}"; exit 1; }
    echo -e "${CYAN}🎯 MODE: Single Image Restoration${NC} — $INPUT"
    echo "========================================"
    python3 src/enhancement_multiagent/pipelines/restoration.py "$INPUT"
    ;;
  augment)
    if [ -z "$INPUT" ]; then echo -e "${RED}❌ Usage: ./run.sh augment <image_path> [weakness]${NC}\n   Weakness: low_light | blur | color_cast | occlusion | rotation | auto"; exit 1; fi
    [ -f "$INPUT" ] || { echo -e "${RED}❌ File not found: $INPUT${NC}"; exit 1; }
    WEAK="${EXTRA:-low_light}"
    echo -e "${CYAN}🧠 MODE: Smart Augmentation${NC} — $INPUT → weakness=$WEAK"
    echo "========================================"
    python3 src/enhancement_multiagent/pipelines/augmentation.py "$INPUT" --weakness "$WEAK" --output-dir restored
    ;;
  benchmark)
    echo -e "${CYAN}🎯 MODE: Benchmark Evaluation${NC}"
    echo "========================================"
    if [ ! -d "benchmark/original" ] || [ -z "$(ls -A benchmark/original 2>/dev/null)" ]; then
      echo ""; echo "📥 Downloading benchmark dataset..."
      python3 scripts/download_benchmark.py
    fi
    echo ""; echo "🏃 Running benchmark..."
    python3 scripts/benchmark.py
    echo ""; echo "📊 Report: reports/benchmark_report.md"
    ;;
  batch)
    if [ -z "$INPUT" ]; then echo -e "${RED}❌ Usage: ./run.sh batch <folder_path>${NC}"; exit 1; fi
    [ -d "$INPUT" ] || { echo -e "${RED}❌ Directory not found: $INPUT${NC}"; exit 1; }
    echo -e "${CYAN}🎯 MODE: Batch Processing${NC} — $INPUT"
    echo "========================================"
    python3 scripts/batch_restore.py "$INPUT"
    ;;
  mcp)
    echo -e "${CYAN}🔌 MODE: MCP Server (stdio)${NC}"
    echo "========================================"
    python3 src/enhancement_multiagent/mcp/server.py
    ;;
  clean)
    echo "🧹 Cleaning generated files..."
    rm -rf restored/* benchmark/restored/* reports/* 2>/dev/null; echo -e "${GREEN}   ✓ Cleaned${NC}"
    ;;
  help|--help|-h)
    echo "Usage: ./run.sh [MODE] [INPUT] [EXTRA]"
    echo ""
    echo "Modes:"
    echo "  demo                    Synthetic demo (restoration)"
    echo "  restore <img>           Restore single image"
    echo "  augment <img> [weak]    Smart augmentation (weakness: low_light|blur|color_cast|occlusion|rotation)"
    echo "  benchmark               Download dataset + full benchmark"
    echo "  batch <folder>          Batch process folder"
    echo "  mcp                     Start MCP tool server"
    echo "  clean                   Remove outputs"
    echo "  help                    This help"
    echo ""
    echo "Examples:"
    echo "  ./run.sh demo"
    echo "  ./run.sh restore my_photo.jpg"
    echo "  ./run.sh augment my_photo.jpg low_light"
    echo "  ./run.sh augment my_photo.jpg blur"
    echo "  ./run.sh benchmark"
    echo "  ./run.sh batch ./vacation_photos/"
    echo ""
    echo "Docs: docs/ARCHITECTURE.md docs/AGENTS.md docs/QUALITY_GATES.md"
    ;;
  *)
    echo -e "${RED}❌ Unknown mode: $MODE${NC}"; echo "   Run './run.sh help'"; exit 1;;
esac

echo ""
echo "========================================"
echo -e "${GREEN}✅ Done!${NC}  ${CYAN}→ restored/ → reports/${NC}"
echo "========================================"
