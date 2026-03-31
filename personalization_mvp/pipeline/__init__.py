"""Product personalization pipeline — MVP."""

from .config import PipelineConfig
from .runner import RunResult, run_pipeline

__all__ = ["PipelineConfig", "RunResult", "run_pipeline"]
