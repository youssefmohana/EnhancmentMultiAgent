#!/usr/bin/env python3
"""
Smart Data Augmentation Pipeline - The vision from the post, live.

Not just flipping images. Planning smarter augmentations that fix what model is bad at,
then validating with Vision LLM quality checks.

Pipeline:
  1. Analyse model's weaknesses (e.g., poor low-light)
  2. Select right agents to target gaps (geometric / photometric / generative)
  3. Generate augmentations (specialised agents)
  4. Validate with extensible quality gates (classical / perceptual / Vision LLM)
  5. Close loop - feed results back & refine strategy
"""
import os
import sys
import argparse
import asyncio
from pathlib import Path
import json
import shutil

# Windows utf-8 fix
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from enhancement_multiagent.agents.orchestrator import MultiAgentOrchestrator
from enhancement_multiagent.planner.augmentation_planner import AugmentationPlanner
from enhancement_multiagent.planner.feedback_loop import FeedbackLoop
from enhancement_multiagent.quality.vision_llm import QualityOrchestrator


async def run_augmentation_pipeline(
    input_path: str,
    output_dir: str = "restored",
    weakness: str = None,
    model_report: str = None,
    image_folder: str = None,
    quality_mode: str = "all",
    max_augmentations: int = 4,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("🧠 Smart Data Augmentation Pipeline")
    print("="*60)
    print(f"📄 Input: {input_path}")
    print(f"📁 Output: {os.path.abspath(output_dir)}")

    # 1. Analyse weaknesses
    planner = AugmentationPlanner()
    if model_report:
        with open(model_report) as f:
            report = json.load(f)
        plan = planner.plan_from_model_report(report, max_augmentations=max_augmentations)
        weaknesses = planner.analyzer.analyze_from_report(report)
        print(f"🔍 Analysed model report: {model_report}")
        print(f"   Weaknesses: {weaknesses}")
    elif image_folder:
        plan = planner.plan_from_dataset(image_folder, performance_json=model_report)
        weaknesses_list = planner.analyzer.analyze_from_folder(image_folder, model_report)
        print(f"🔍 Analysed dataset: {image_folder} -> {weaknesses_list}")
        weaknesses = weaknesses_list
    elif weakness:
        print(f"🔍 Targeting weakness: {weakness}")
        weaknesses = [{"type": weakness, "severity": 0.9}]
        plan = planner.plan_for_weaknesses(weaknesses, max_augmentations=max_augmentations)
    else:
        # Auto-diagnose from image itself via orchestrator
        orchestrator_diag = MultiAgentOrchestrator()
        plan = orchestrator_diag.diagnose_and_plan(input_path, weakness_hint=None)
        weaknesses = [{"type": "auto_detected", "severity": 0.6}]
        print(f"🔍 Auto-diagnosed plan: {plan}")

    print(f"📋 Plan ({len(plan)} steps):")
    for i, s in enumerate(plan,1):
        print(f"   {i}. {s['agent']}::{s['operation']} {s.get('params',{})}")

    # 2 & 3. Generate augmentations via agents
    orchestrator = MultiAgentOrchestrator()
    quality = QualityOrchestrator()

    results = []
    # For single input image: each plan step generates a separate augmented version (alternative view)
    # Also demonstrate chaining: first as chain example
    print("\n⚡ Generating augmentations...")

    base_name = Path(input_path).stem
    suffix = Path(input_path).suffix or ".png"

    for idx, step in enumerate(plan):
        agent = step["agent"]
        op = step["operation"]
        params = step.get("params", {})
        out_path = os.path.join(output_dir, f"{base_name}_aug{idx}_{agent}_{op}{suffix}")
        # Need a temp input for each: use original as base for each augmentation (not chain) for diversity evaluation
        # We expose both modes: single-step augmentations for training data expansion
        res = orchestrator.execute(input_path, out_path, agent, op, **params)
        if not res.success:
            print(f"   ❌ {agent}::{op} failed: {res.message}")
            continue
        # 4. Validate with quality gates
        q_result = quality.validate(input_path, out_path, mode=quality_mode)
        status = "✅ keep" if q_result["final_pass"] else "⚠️ reject"
        print(f"   {idx+1}. {agent}::{op} -> {out_path} | quality {q_result['avg_score']:.3f} {status} | gates {q_result['votes']}")
        results.append({"agent": agent, "operation": op, "params": params, "output": out_path, "quality": q_result, "success": True})

    # 5. Close loop - feedback
    feedback = FeedbackLoop()
    feedback.log_cycle(weaknesses, plan, results)
    print("\n🔄 Feedback loop logged:")
    print(f"   {feedback.report()}")
    print(f"   Summary: {feedback.history[-1]['summary'] if feedback.history else {}}")

    # Also demonstrate chained execution (pipeline style)
    if len(plan) >= 2:
        chained_out = os.path.join(output_dir, f"{base_name}_chained{suffix}")
        # Chain first 2 steps as example
        chained_results = orchestrator.execute_plan(input_path, plan[:2], output_dir=output_dir)
        print(f"\n🔗 Chained example (first 2 steps): {chained_results[-1].output_path if chained_results else 'none'}")

    # Write manifest
    manifest_path = os.path.join(output_dir, f"{base_name}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"input": input_path, "weaknesses": weaknesses, "plan": plan, "results": results}, f, indent=2)
    print(f"\n📝 Manifest saved: {manifest_path}")
    print("="*60)
    print("✅ Pipeline complete - augmentation data ready for training")
    print("="*60)
    return results


def main():
    parser = argparse.ArgumentParser(description="Smart Data Augmentation Pipeline")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--output-dir", default="restored", help="Output directory")
    parser.add_argument("--weakness", default=None, help="Target weakness e.g. low_light, blur, occlusion, color_cast, rotation")
    parser.add_argument("--model-report", default=None, help="JSON with per_class_accuracy to auto-plan")
    parser.add_argument("--image-folder", default=None, help="Dataset folder to analyze distribution gaps")
    parser.add_argument("--quality", default="all", choices=["all","classical","perceptual","vision"], help="Quality gate mode")
    parser.add_argument("--max-augmentations", type=int, default=4, help="Max augmentations to generate")
    args = parser.parse_args()
    asyncio.run(run_augmentation_pipeline(args.input, args.output_dir, args.weakness, args.model_report, args.image_folder, args.quality, args.max_augmentations))


if __name__ == "__main__":
    main()
