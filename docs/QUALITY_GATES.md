# 🔍 Quality Gates — Extensible Validation

> Swap between classical metrics, learned perceptual metrics, or Vision LLM reasoning.

| Gate | File | Metrics | When to Use |
|------|------|---------|-------------|
| **Classical** | `quality/classical.py` | PSNR, SSIM, MSE, blur, brightness | Fast, deterministic CI |
| **Perceptual** | `quality/perceptual.py` | Edge IoU, hist correlation, LPIPS proxy | Perceptual similarity |
| **Vision LLM** | `quality/vision_llm.py` | `semantic_valid`, `realism_score`, `artifact_severity` | Semantic sense check |

## 🧠 Vision LLM Oracle (Central Quality Oracle)
- Model: auto-detects `llava`, `llama3.2-vision`, `moondream` via Ollama
- Fallback: text LLM reasoning + heuristic (SSIM/brightness) → pipeline never blocks offline
- Prompt asks JSON: `{semantic_valid, artifact_severity, realism_score, keep}`

```python
from enhancement_multiagent.quality.vision_llm import QualityOrchestrator
q = QualityOrchestrator()  # [Classical, Perceptual, VisionLLM]
res = q.validate("orig.png", "aug.png", mode="all")  # or "classical" / "vision" / ["classical","vision"]
# res = {"final_pass": bool, "avg_score": float, "votes": "2/3", "gates": {...}}
```

## 🔄 Mode
- `all` — all gates, majority + Vision LLM weight
- `classical` — fast for training loops
- `vision` — LLM only (requires Ollama)
- `["classical","perceptual"]` — custom

## ➕ Custom Gate
```python
from enhancement_multiagent.quality.base import QualityGate, QualityResult
class MyGate(QualityGate):
    def __init__(self): super().__init__("my_gate", threshold=0.7)
    def validate(self, orig, aug): return QualityResult("my_gate", passed=True, score=0.9, details={})
    def describe(self): return {"name": self.name}
```
