import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from enhancement_multiagent.planner.augmentation_planner import AugmentationPlanner

planner = AugmentationPlanner()
report = {"per_class_accuracy": {"low_light": 0.52, "normal": 0.92}}
weak = planner.analyzer.analyze_from_report(report)
print("Weakness:", weak)
plan = planner.plan_for_weaknesses(weak)
print("Plan:", plan)
assert len(plan) > 0
# Vision loop
from enhancement_multiagent.planner.feedback_loop import FeedbackLoop
loop = FeedbackLoop(log_path="tmp/feedback.json")
loop.log_cycle(weak, plan, [{"agent":"photometric","quality":{"final_pass":True}}])
print("[PASS] planner + feedback")
