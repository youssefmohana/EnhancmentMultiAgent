"""Feedback loop - self-improving augmentation that gets smarter every cycle."""
from typing import List, Dict, Any
import json
import os
from pathlib import Path
from datetime import datetime


class FeedbackLoop:
    """
    Closes the loop: validate -> feed results back -> refine strategy.
    Gets smarter with every training cycle.
    """

    def __init__(self, log_path: str = "reports/feedback_log.json"):
        self.log_path = log_path
        self.history: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path) as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []

    def _save(self):
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def log_cycle(self, weaknesses: List[Dict], plan: List[Dict], results: List[Dict], model_metrics: Dict = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "weaknesses": weaknesses,
            "plan": plan,
            "results": results,
            "model_metrics": model_metrics,
            "summary": self._summarize(results)
        }
        self.history.append(entry)
        self._save()
        return entry

    def _summarize(self, results: List[Dict]) -> Dict[str, Any]:
        total = len(results)
        passed = sum(1 for r in results if r.get("quality", {}).get("final_pass", False))
        failed = total - passed
        by_agent = {}
        for r in results:
            agent = r.get("agent", "unknown")
            by_agent[agent] = by_agent.get(agent, 0) + 1
        return {"total": total, "passed": passed, "failed": failed, "pass_rate": round(passed/max(total,1),3), "by_agent": by_agent}

    def get_refinement_hint(self) -> str:
        """Analyze history to suggest next strategy adjustment."""
        if not self.history:
            return "default"
        last = self.history[-1]
        rate = last["summary"]["pass_rate"]
        if rate < 0.5:
            return "conservative"  # planner will use safer ops
        # Check which agent failed most
        return "default"

    def report(self) -> str:
        if not self.history:
            return "No cycles logged yet."
        last = self.history[-1]
        return f"Cycles: {len(self.history)} | Last pass rate: {last['summary']['pass_rate']*100:.1f}% | History logged to {self.log_path}"
