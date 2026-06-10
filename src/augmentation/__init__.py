"""Augmentation policies for detector-agnostic anomaly-transfer experiments."""

from .policies import AugmentationResult, build_augmentation_result

__all__ = ["AugmentationResult", "build_augmentation_result"]
