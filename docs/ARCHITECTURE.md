# 🏗️ Architecture — Enhancement MultiAgent

## 📐 Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                        User Image / Dataset                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
               ┌───────────────▼───────────────┐
               │   🧩 MCP Image Tool Server    │
               │  (OpenCV · FastMCP · stdio)   │
               │  • analyze_image_quality     │
               │  • flip / rotate / crop      │
               │  • brightness / color_jitter │
               │  • synthetic_occlusion etc.  │
               └───────────────┬───────────────┘
                               │ MCP protocol
               ┌───────────────▼───────────────┐
               │ 🎛️ Multi-Agent Orchestrator   │
               │  ├─ Geometric Agent          │
               │  ├─ Photometric Agent        │
               │  └─ Generative Agent         │
               └───────────────┬───────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │        🧠 Data Augmentation Planner         │
        │  1. Weakness Analyzer  → detect gaps       │
        │  2. Strategy Selector  → pick agents       │
        │  3. Execution          → generate augs     │
        │  4. Quality Gates      → validate          │
        │  5. Feedback Loop      → refine strategy   │
        └──────────────────────┬──────────────────────┘
                               │
               ┌───────────────▼───────────────┐
               │ 🔍 Quality Gates (Extensible) │
               │  • Classical (PSNR/SSIM)      │
               │  • Perceptual (Edge/Hist)     │
               │  • Vision LLM Oracle (Ollama) │
               └───────────────┬───────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │ 📊 Benchmarking + Integrations              │
        │  • PyTorch · TensorFlow (dev) · HF (dev)   │
        │  • Reports: markdown / csv / charts        │
        └─────────────────────────────────────────────┘
```

## 📦 Layers
| Layer | Path | Responsibility |
|-------|------|----------------|
| **Agents** | `src/enhancement_multiagent/agents/` | Specialized augmenters, plug-in without core changes |
| **Quality** | `src/enhancement_multiagent/quality/` | Swappable gates, Vision LLM as oracle |
| **Planner** | `src/enhancement_multiagent/planner/` | Model-aware strategy & feedback |
| **MCP** | `src/enhancement_multiagent/mcp/` | Tool server, any agent discovers tools |
| **Pipelines** | `src/enhancement_multiagent/pipelines/` | `augmentation.py`, `restoration.py` |
| **Benchmarking** | `src/enhancement_multiagent/benchmarking/` | Metrics + reporting |
| **Integrations** | `src/enhancement_multiagent/integrations/` | Framework hooks |
| **Scripts** | `scripts/` | CLI entry points (legacy shims at root for compat) |

## 🔄 Data Flow
`Analyse weaknesses → Plan with right agents → Generate augmentations → Validate with gates → Close loop → Feed back`

## 🔌 Extensibility
- Add agent: subclass `BaseAgent` in `agents/custom.py` and `orchestrator.register_agent("custom", CustomAgent())`
- Add gate: subclass `QualityGate` in `quality/custom.py` and pass to `QualityOrchestrator([CustomGate(), ...])`
