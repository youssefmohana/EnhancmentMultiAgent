"""
Enhancement MultiAgent - Smart Data Augmentation System

Everyone's augmenting data. Almost no one's asking if it's actually good data.
This package implements a multi-agent system that generates *smarter* augmentations
targeted at model weaknesses, validated by extensible quality gates including Vision LLM.
"""

__version__ = "0.2.0"
__author__ = "youssefmohana"

from .agents.orchestrator import MultiAgentOrchestrator
from .planner.augmentation_planner import AugmentationPlanner
from .quality.base import QualityGate

__all__ = ["MultiAgentOrchestrator", "AugmentationPlanner", "QualityGate"]
