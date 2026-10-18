"""Breadboard vision registration, quality gates and pose inference."""
from .registration import (BreadboardRegistration, QualityGateError, QualityReport,
                           RegistrationResult, generate_synthetic_aruco_board_image)
from .detector import BreadboardPoseDetector
from .classical_detector import ClassicalBreadboardDetector, BaselineMismatchError

__all__ = ['BreadboardRegistration', 'QualityGateError', 'QualityReport',
           'RegistrationResult', 'generate_synthetic_aruco_board_image',
           'BreadboardPoseDetector', 'ClassicalBreadboardDetector', 'BaselineMismatchError']
