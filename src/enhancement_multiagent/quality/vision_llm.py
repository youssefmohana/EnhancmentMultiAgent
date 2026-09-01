"""Vision LLM oracle - teaches system to see if augmented image makes sense."""
import base64
import json
import os
from typing import Dict, Any
from .base import QualityGate, QualityResult


class VisionLLMOracle(QualityGate):
    """
    Central quality oracle using Vision LLM (Ollama llava / llama3.2-vision).
    Reasoning: asks LLM to judge semantic validity, not just pixels.
    Falls back to heuristic if Ollama not available -> system still works.
    """

    def __init__(self, model: str = "llava", threshold: float = 0.6, ollama_host: str = "http://localhost:11434"):
        super().__init__(name="vision_llm", threshold=threshold)
        self.model = model
        self.host = ollama_host
        # Try to detect available vision model
        self.available_models = self._detect_models()

    def _detect_models(self):
        try:
            import ollama
            models = ollama.list()
            # ollama 0.3 returns dict with 'models' key
            if isinstance(models, dict) and "models" in models:
                return [m.get("name", "") for m in models["models"]]
            if hasattr(models, "models"):
                return [m.model for m in models.models]
            return []
        except Exception:
            return []

    def _select_model(self):
        # Prefer vision models if available
        for cand in ["llava", "llava:13b", "llama3.2-vision", "bakllava", "moondream"]:
            for m in self.available_models:
                if cand in m:
                    return m
        # fallback to text model for reasoning with pseudo-analysis
        if self.available_models:
            return self.available_models[0]
        return self.model

    def describe(self):
        return {"name": self.name, "model": self._select_model(), "capabilities": ["semantic_validity", "artifact_detection", "realism_score"], "threshold": self.threshold}

    def _encode_image(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def validate(self, original_path: str, augmented_path: str) -> QualityResult:
        # Attempt Vision LLM reasoning
        prompt = """You are a quality oracle for data augmentation.
Compare original and augmented images (augmented is provided). Answer JSON only:
{
  \"semantic_valid\": bool,  # does augmented image still look realistic and label-preserving?
  \"artifact_severity\": 0-10,  # 0 none, 10 severe artifacts
  \"realism_score\": 0-10,  # 0 unrealistic, 10 perfect
  \"issues\": [\"list any artifacts: color banding, black borders, unnatural blur, etc\"],
  \"keep\": bool  # should we keep this augmentation for training?
}
Consider: flipping is valid, brightness/gamma shifts are valid if not extreme, heavy blur/noise may degrade but still valid if label preserved. Reject only if image is corrupted, completely black/white, or has severe artifacts."""

        try:
            # Check if Ollama is reachable
            import ollama
            model = self._select_model()
            # Check if vision capable
            is_vision = any(v in model for v in ["llava", "vision", "moondream", "bakllava"])
            if is_vision and os.path.exists(augmented_path):
                try:
                    # Use ollama chat with image
                    resp = ollama.chat(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt, "images": [self._encode_image(augmented_path)]}
                        ],
                        format="json",
                        options={"temperature": 0.1}
                    )
                    content = resp.get("message", {}).get("content", "") if isinstance(resp, dict) else getattr(resp.message, "content", "")
                    data = json.loads(content) if content else {}
                    semantic_valid = bool(data.get("semantic_valid", True))
                    realism = float(data.get("realism_score", 7)) / 10.0
                    artifact = float(data.get("artifact_severity", 3)) / 10.0
                    keep = bool(data.get("keep", True))
                    issues = data.get("issues", [])
                    score = (realism*0.6 + (1-artifact)*0.4)
                    passed = keep and score >= self.threshold and semantic_valid
                    details = {"model": model, "semantic_valid": semantic_valid, "realism": realism, "artifact_severity": artifact, "issues": issues, "raw": data}
                    msg = f"VisionLLM {model}: realism {realism:.2f} keep={keep} issues={issues}"
                    return QualityResult(self.name, passed, score, details, msg, self.threshold)
                except Exception as e:
                    # Fall back to heuristic below
                    pass

            # Text-only LLM fallback: describe metrics and ask for reasoning (without image)
            # Provide heuristic analysis to LLM
            try:
                import cv2
                import numpy as np
                from skimage.metrics import structural_similarity as ssim
                orig = cv2.imread(original_path)
                aug = cv2.imread(augmented_path)
                details_for_llm = {}
                if orig is not None and aug is not None:
                    if orig.shape != aug.shape:
                        aug_r = cv2.resize(aug, (orig.shape[1], orig.shape[0]))
                    else:
                        aug_r = aug
                    gray_o = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
                    gray_a = cv2.cvtColor(aug_r, cv2.COLOR_BGR2GRAY)
                    ssim_val = float(ssim(gray_o, gray_a, data_range=255))
                    brightness_a = float(gray_a.mean())
                    details_for_llm = {"ssim": ssim_val, "brightness_a": brightness_a, "shape_orig": orig.shape, "shape_aug": aug.shape}
                heuristic_prompt = f"{prompt}\n\nHeuristic metrics (since vision not available): {json.dumps(details_for_llm)}\nAssume augmented image is available with those metrics. Decide keep/semantic_valid based on thresholds: ssim>0.5 valid, brightness 20-235 valid."
                resp = ollama.chat(model=model, messages=[{"role": "user", "content": heuristic_prompt}], format="json", options={"temperature": 0.1})
                content = resp.get("message", {}).get("content", "") if isinstance(resp, dict) else getattr(resp.message, "content", "")
                data = json.loads(content) if content else {}
                if data:
                    semantic_valid = bool(data.get("semantic_valid", True))
                    realism = float(data.get("realism_score", 7)) / 10.0
                    artifact = float(data.get("artifact_severity", 3)) / 10.0
                    keep = bool(data.get("keep", True))
                    score = (realism*0.6 + (1-artifact)*0.4)
                    passed = keep and score >= self.threshold
                    return QualityResult(self.name, passed, score, {"model": model, "fallback": "text_llm", "raw": data}, f"LLM text fallback {model}: keep={keep}", self.threshold)
            except Exception:
                pass

        except ImportError:
            pass
        except Exception:
            pass

        # Final heuristic fallback - still provides a quality decision when Ollama unavailable
        # This ensures pipeline works offline / CI
        try:
            import cv2
            import numpy as np
            from skimage.metrics import structural_similarity as ssim
            orig = cv2.imread(original_path)
            aug = cv2.imread(augmented_path)
            if orig is None or aug is None:
                return QualityResult(self.name, False, 0.0, {"fallback": "heuristic"}, "Fallback: load failed", self.threshold)
            if orig.shape != aug.shape:
                aug = cv2.resize(aug, (orig.shape[1], orig.shape[0]))
            gray_o = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            gray_a = cv2.cvtColor(aug, cv2.COLOR_BGR2GRAY)
            ssim_val = float(ssim(gray_o, gray_a, data_range=255))
            brightness = float(gray_a.mean())
            # Simple rules: reject if degenerate
            if brightness < 8 or brightness > 248:
                return QualityResult(self.name, False, 0.1, {"fallback": "heuristic", "ssim": ssim_val, "brightness": brightness}, "Fallback: degenerate brightness", self.threshold)
            if ssim_val < 0.25:
                return QualityResult(self.name, False, ssim_val, {"fallback": "heuristic", "ssim": ssim_val}, "Fallback: structure destroyed", self.threshold)
            score = max(0, min(1, ssim_val))
            passed = score >= 0.35
            return QualityResult(self.name, passed, score, {"fallback": "heuristic", "ssim": ssim_val, "brightness": brightness, "model": "none"}, f"Fallback heuristic ssim {ssim_val:.3f}", self.threshold)
        except Exception as e:
            return QualityResult(self.name, True, 0.5, {"fallback": "heuristic_failed", "error": str(e)}, "Fallback: assumed pass (heuristic error)", self.threshold)


class QualityOrchestrator:
    """Extensible orchestrator to swap between classical / perceptual / Vision LLM gates."""

    def __init__(self, gates=None):
        if gates is None:
            # Lazy imports to avoid circular
            from .classical import ClassicalGate
            from .perceptual import PerceptualGate
            gates = [ClassicalGate(), PerceptualGate(), VisionLLMOracle()]
        self.gates = gates

    def validate(self, original_path: str, augmented_path: str, mode: str = "all") -> Dict[str, Any]:
        """
        mode: 'all' runs all gates, 'classical', 'perceptual', 'vision' or custom list.
        Returns aggregated decision.
        """
        results = {}
        # Filter gates by mode
        active = self.gates
        if mode == "classical":
            active = [g for g in self.gates if g.name == "classical"]
        elif mode == "perceptual":
            active = [g for g in self.gates if g.name == "perceptual"]
        elif mode == "vision":
            active = [g for g in self.gates if g.name == "vision_llm"]
        elif isinstance(mode, list):
            active = [g for g in self.gates if g.name in mode]

        passed_all = True
        avg_score = 0.0
        for gate in active:
            res = gate.validate(original_path, augmented_path)
            results[gate.name] = {"passed": res.passed, "score": res.score, "details": res.details, "message": res.message}
            passed_all = passed_all and res.passed
            avg_score += res.score
        avg_score = avg_score / max(len(active),1)
        # Majority or strict? For augmentation we use majority: need 2/3 gates to pass OR vision_llm pass alone if present
        votes = sum(1 for r in results.values() if r["passed"])
        # If vision_llm says keep, we trust it higher weight
        if "vision_llm" in results and results["vision_llm"]["passed"] and avg_score >= 0.45:
            final_pass = True
        else:
            final_pass = votes >= (len(active)/2)

        return {
            "final_pass": final_pass,
            "avg_score": round(avg_score, 4),
            "votes": f"{votes}/{len(active)}",
            "gates": results,
            "mode": mode
        }

    def list_gates(self):
        return [g.describe() for g in self.gates]
