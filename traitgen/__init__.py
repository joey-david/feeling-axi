"""Concept-general adaptation layer for the released Pain-axis pipeline."""

from .specs import TraitSpec, load_spec
from .validate import validate_core_dataset, validate_screen_dataset

__all__ = ["TraitSpec", "load_spec", "validate_core_dataset", "validate_screen_dataset"]
