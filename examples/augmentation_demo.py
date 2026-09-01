"""Demo: smart augmentation pipeline from post's vision."""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from enhancement_multiagent.planner.augmentation_planner import AugmentationPlanner
from enhancement_multiagent.quality.vision_llm import QualityOrchestrator
from enhancement_multiagent.agents.orchestrator import MultiAgentOrchestrator

async def main():
    # 1. Model says it fails on low-light
    report = {"per_class_accuracy": {"low_light": 0.52, "normal": 0.91, "blurry": 0.61}}
    planner = AugmentationPlanner()
    weaknesses = planner.analyzer.analyze_from_report(report)
    print("Weaknesses:", weaknesses)
    plan = planner.plan_for_weaknesses(weaknesses)
    print("Plan:", plan)
    # 2. Apply to sample image (create dummy)
    import cv2, numpy as np, pathlib
    pathlib.Path("demo_images").mkdir(exist_ok=True)
    img = np.ones((256,256,3), dtype=np.uint8)*180
    cv2.circle(img, (128,128), 60, (80,80,200), -1)
    cv2.imwrite("demo_images/sample.png", img)
    orch = MultiAgentOrchestrator()
    print("Available agents:", orch.list_agents().keys())
    results = orch.execute_plan("demo_images/sample.png", plan[:2], output_dir="restored")
    for r in results:
        print(r)
    q = QualityOrchestrator()
    print("Quality:", q.validate("demo_images/sample.png", results[-1].output_path if results else "demo_images/sample.png"))

if __name__ == "__main__":
    asyncio.run(main())
