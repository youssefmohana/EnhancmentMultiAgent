import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
import cv2, numpy as np, tempfile, pathlib
from enhancement_multiagent.agents.orchestrator import MultiAgentOrchestrator

def test_geometric_flip():
    pathlib.Path("tmp").mkdir(exist_ok=True)
    img = np.ones((64,64,3), dtype=np.uint8)*100
    cv2.imwrite("tmp/test_in.png", img)
    orch = MultiAgentOrchestrator()
    res = orch.execute("tmp/test_in.png", "tmp/test_out.png", "geometric", "flip", direction="horizontal")
    assert res.success and os.path.exists("tmp/test_out.png")
    print("[PASS] geometric flip")

def test_photometric_brightness():
    orch = MultiAgentOrchestrator()
    res = orch.execute("tmp/test_in.png", "tmp/test_out2.png", "photometric", "brightness", gamma=0.6)
    assert res.success
    print("[PASS] photometric brightness")

def test_orchestrator_plan():
    orch = MultiAgentOrchestrator()
    plan = orch.diagnose_and_plan("tmp/test_in.png", weakness_hint="low_light")
    assert len(plan) > 0
    print("[PASS] orchestrator plan", plan)

if __name__ == "__main__":
    test_geometric_flip()
    test_photometric_brightness()
    test_orchestrator_plan()
    print("All agent tests passed")
