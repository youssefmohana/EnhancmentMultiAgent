"""Base quality gate - extensible design to swap metrics."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class QualityResult:
    gate_name: str
    passed: bool
    score: float  # 0-1 or metric specific, normalized where possible
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    threshold: float = 0.0


class QualityGate(ABC):
    """Abstract quality gate. Extend to add new metrics (classical, learned, Vision LLM)."""

    def __init__(self, name: str, threshold: float = 0.5):
        self.name = name
        self.threshold = threshold

    @abstractmethod
    def validate(self, original_path: str, augmented_path: str) -> QualityResult:
        pass

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        pass
