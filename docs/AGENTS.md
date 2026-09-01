# 🤖 Agents — Modular by Design

> No more monolithic scripts. Each task has a specialist.

| Agent | File | Ops | Use Case |
|-------|------|-----|----------|
| **Geometric** | `agents/geometric.py` | `flip`, `rotate`, `scale`, `crop`, `affine`, `perspective` | Rotation/scale invariance, viewpoint robustness |
| **Photometric** | `agents/photometric.py` | `brightness`, `contrast`, `color_jitter`, `blur`, `noise`, `clahe` | Low-light, color cast, sensor noise |
| **Generative** | `agents/generative.py` | `upscale`, `inpaint`, `synthetic_occlusion`, `cutmix`, `elastic` | Occlusion, super-resolution, generative CV |

## 🧩 Base Interface
```python
from enhancement_multiagent.agents.base import BaseAgent, AgentResult

class MyAgent(BaseAgent):
    def __init__(self): super().__init__("my_agent", "does X")
    def augment(self, input_path, output_path, operation="my_op", **kw) -> AgentResult: ...
    def get_available_operations(self): return [...]
```

## 🔌 Plug-in Example
```python
from enhancement_multiagent.agents.orchestrator import MultiAgentOrchestrator
from my_agents import MyAgent

orch = MultiAgentOrchestrator()
orch.register_agent("my", MyAgent())  # no core changes
orch.execute("in.png", "out.png", "my", "my_op", param=1)
```

## 🎯 Orchestrator
`MultiAgentOrchestrator` chains plans:
```python
plan = [
  {"agent":"geometric","operation":"flip","params":{"direction":"horizontal"}},
  {"agent":"photometric","operation":"brightness","params":{"gamma":0.5}},
]
orch.execute_plan("input.jpg", plan, output_dir="restored")
```
Auto-diagnosis: `orch.diagnose_and_plan("image.jpg", weakness_hint="low_light")`
