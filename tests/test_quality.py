import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
import cv2, numpy as np
from enhancement_multiagent.quality.vision_llm import QualityOrchestrator

# create two images
import pathlib
pathlib.Path("tmp").mkdir(exist_ok=True)
a = np.ones((64,64,3), dtype=np.uint8)*120
b = np.ones((64,64,3), dtype=np.uint8)*140
cv2.imwrite("tmp/a.png", a)
cv2.imwrite("tmp/b.png", b)
q = QualityOrchestrator()
res = q.validate("tmp/a.png", "tmp/b.png", mode="all")
print("Quality result:", res)
assert "final_pass" in res
assert "gates" in res
print("[PASS] quality orchestrator")
