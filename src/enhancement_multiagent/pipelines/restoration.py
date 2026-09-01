#!/usr/bin/env python3
"""
Image Restoration Orchestrator - backward compatible + new augmentation pipeline wrapper.
Provides restore_image() for benchmark.py while delegating to new MultiAgentOrchestrator.

This file bridges the original restoration system with the new smart augmentation system.
"""
import os
import sys
import asyncio
import json
import cv2
import numpy as np
from pathlib import Path

from enhancement_multiagent.agents.orchestrator import MultiAgentOrchestrator
from enhancement_multiagent.quality.vision_llm import QualityOrchestrator

# Keep legacy MCP tool logic available for direct calls (optional)
try:
    from enhancement_multiagent.mcp.server import analyze_image_quality as mcp_analyze
except Exception:
    try:
        from mcp_image_server import analyze_image_quality as mcp_analyze  # fallback for legacy shim
    except Exception:
        mcp_analyze = None


async def restore_image(image_path: str, output_path: str = None) -> str:
    """
    Restore degraded image using multi-agent pipeline.
    Compatible with benchmark.py: await restore_image(degraded_path) -> restored_path

    New behavior: uses Orchestrator Diagnose->Plan->Execute with quality validation.
    Falls back to simple OpenCV chain if LLM unavailable.
    """
    if output_path is None:
        base = Path(image_path).stem
        suffix = Path(image_path).suffix
        output_path = str(Path("restored") / f"{base}_restored{suffix}")
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)

    orchestrator = MultiAgentOrchestrator()
    quality = QualityOrchestrator()

    # Diagnose and plan restoration steps (treat restoration as targeted augmentation)
    # Use weakness hint auto-detected from image quality
    plan = orchestrator.diagnose_and_plan(image_path)

    # If no plan (good image), return copy
    if not plan:
        import shutil
        shutil.copy(image_path, output_path)
        return output_path

    # Map diagnosis to restoration ops (brightness/contrast/sharpen etc.)
    # The orchestrator's plan already uses correct agents
    # Execute chain
    results = orchestrator.execute_plan(image_path, plan, output_dir=os.path.dirname(output_path) or "restored")

    # Get final output (last successful)
    final = None
    for r in reversed(results):
        if r.success and os.path.exists(r.output_path):
            final = r.output_path
            break
    if final is None:
        # fallback: simple enhance
        img = cv2.imread(image_path)
        if img is not None:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l,a,b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            result = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
            cv2.imwrite(output_path, result)
            return output_path
        # copy as last resort
        import shutil
        shutil.copy(image_path, output_path)
        return output_path

    # If final is not the requested output_path, copy it
    if final != output_path:
        import shutil
        shutil.copy(final, output_path)

    # Optional: quality validation (non-blocking for restoration, just log)
    try:
        q = quality.validate(image_path, output_path, mode="classical")
        # print for debugging but don't fail restoration
        # print(f"Restoration quality: {q}")
        pass
    except Exception:
        pass

    return output_path


async def restore_image_smart(image_path: str, output_path: str = None, weakness: str = None) -> str:
    """Smart variant that explicitly targets a weakness via augmentation planner."""
    from enhancement_multiagent.planner.augmentation_planner import AugmentationPlanner
    planner = AugmentationPlanner()
    if output_path is None:
        output_path = str(Path("restored") / f"{Path(image_path).stem}_restored{Path(image_path).suffix}")
    plan = planner.plan_from_hint(weakness or "low_light")
    orchestrator = MultiAgentOrchestrator()
    results = orchestrator.execute_plan(image_path, plan[:1], output_dir=os.path.dirname(output_path) or "restored")
    if results and results[-1].success:
        import shutil
        shutil.copy(results[-1].output_path, output_path)
        return output_path
    return await restore_image(image_path, output_path)


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: enhance-restore <image_path> [output_path]  OR  python -m enhancement_multiagent.pipelines.restoration <image_path]")
        # demo: create synthetic degraded image
        demo_dir = Path("demo_images")
        demo_dir.mkdir(exist_ok=True)
        # generate synthetic degraded
        img = np.random.randint(180, 220, (256,256,3), dtype=np.uint8)
        cv2.rectangle(img, (50,50), (200,200), (40,40,120), -1)
        demo_path = str(demo_dir / "demo_input.png")
        cv2.imwrite(demo_path, img)
        print(f"Created demo image: {demo_path}")
        out = asyncio.run(restore_image(demo_path))
        print(f"Restored: {out}")
    else:
        inp = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else None
        result = asyncio.run(restore_image(inp, out))
        print(f"Restored -> {result}")


if __name__ == "__main__":
    main()
