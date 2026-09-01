"""Base agent interface - plug in new agents without touching core system."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List
import time
import cv2
import numpy as np


@dataclass
class AgentResult:
    success: bool
    output_path: str
    agent_name: str
    operation: str
    params: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base for all enhancement agents. Modular by design."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.capabilities: List[str] = []

    @abstractmethod
    def augment(self, input_path: str, output_path: str, **kwargs) -> AgentResult:
        """Apply augmentation / enhancement to image. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_available_operations(self) -> List[Dict[str, Any]]:
        """Return list of ops this agent can perform with param specs."""
        pass

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return img

    def _save_image(self, img: np.ndarray, path: str):
        cv2.imwrite(path, img)

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name}>"
