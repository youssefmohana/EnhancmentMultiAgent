"""Multi-Agent Orchestrator - specialized agents working together."""
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from .base import AgentResult
from .geometric import GeometricAgent
from .photometric import PhotometricAgent
from .generative import GenerativeAgent


class MultiAgentOrchestrator:
    """
    Orchestrates specialized enhancement agents.
    Each task is handled by the right agent - no monolithic scripts.
    Agents are plug-and-play: add new agents without touching core.
    """

    def __init__(self):
        self.agents = {
            "geometric": GeometricAgent(),
            "photometric": PhotometricAgent(),
            "generative": GenerativeAgent(),
        }
        # Keep restoration tools as fallback agents (uses photometric/generative internally)
        self.history: List[AgentResult] = []

    def list_agents(self) -> Dict[str, Any]:
        return {name: {"description": a.description, "capabilities": a.capabilities, "ops": a.get_available_operations()} for name, a in self.agents.items()}

    def register_agent(self, name: str, agent):
        """Plug in new agent without touching core system."""
        self.agents[name] = agent

    def execute(self, input_path: str, output_path: str, agent_name: str, operation: str, **params) -> AgentResult:
        if agent_name not in self.agents:
            return AgentResult(False, output_path, agent_name, operation, params, 0, f"Agent {agent_name} not found. Available: {list(self.agents.keys())}")
        result = self.agents[agent_name].augment(input_path, output_path, operation=operation, **params)
        self.history.append(result)
        return result

    def execute_plan(self, input_path: str, plan: List[Dict[str, Any]], output_dir: str = "restored") -> List[AgentResult]:
        """
        Execute a plan: [{'agent': 'photometric', 'operation': 'brightness', 'params': {'gamma':0.6}}, ...]
        Chains augmentations sequentially, feeding output of one as input to next.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        current_input = input_path
        results: List[AgentResult] = []
        for idx, step in enumerate(plan):
            agent = step.get("agent", "photometric")
            op = step.get("operation", "brightness")
            params = step.get("params", {})
            # intermediate path
            stem = Path(input_path).stem
            suffix = Path(input_path).suffix
            is_last = idx == len(plan)-1
            out_path = f"{output_dir}/{stem}_aug_{idx}_{agent}_{op}{suffix}" if not is_last else f"{output_dir}/{stem}_final{suffix}"
            # For chained, use previous output as input
            res = self.execute(current_input, out_path, agent, op, **params)
            results.append(res)
            if not res.success:
                break
            current_input = out_path
        return results

    def diagnose_and_plan(self, image_path: str, weakness_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Simple rule-based diagnosis (fallback when LLM not available).
        Vision LLM will replace this reasoning in full integration.
        Returns a plan tailored to weakness_hint or auto-detected issues.
        """
        # Try to analyze image quality using classical metrics if available
        plan = []
        try:
            import cv2
            import numpy as np
            img = cv2.imread(image_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness = float(gray.mean())
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                contrast = float(gray.std())
                # heuristics
                if weakness_hint == "low_light" or brightness < 60:
                    plan.append({"agent": "photometric", "operation": "brightness", "params": {"gamma": 0.6}})
                    plan.append({"agent": "photometric", "operation": "contrast", "params": {"clip_limit": 2.5}})
                if weakness_hint == "blur" or lap_var < 100:
                    plan.append({"agent": "photometric", "operation": "sharpen", "params": {}})
                if weakness_hint == "low_contrast" or contrast < 30:
                    plan.append({"agent": "photometric", "operation": "contrast", "params": {"clip_limit": 2.0}})
                if weakness_hint == "rotation":
                    plan.append({"agent": "geometric", "operation": "rotate", "params": {"angle": 15}})
                if weakness_hint == "scale":
                    plan.append({"agent": "geometric", "operation": "scale", "params": {"factor": 1.2}})
        except Exception:
            pass
        # fallback: if no weakness detected, propose diverse augmentation set
        if not plan:
            if weakness_hint == "occlusion":
                plan = [{"agent": "generative", "operation": "synthetic_occlusion", "params": {"count": 2}}]
            elif weakness_hint == "color_cast":
                plan = [{"agent": "photometric", "operation": "color_jitter", "params": {"hue": 8, "saturation": 1.3}}]
            else:
                # generic smart augmentation: geometric + photometric
                plan = [
                    {"agent": "geometric", "operation": "flip", "params": {"direction": "horizontal"}},
                    {"agent": "photometric", "operation": "brightness", "params": {"gamma": 0.7}},
                ]
        return plan
