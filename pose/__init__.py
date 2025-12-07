"""Pose Detection Module."""

from .base_detector import BaseDetector, BaseBodyDetector, BaseHandDetector, BaseFaceDetector
from .detector_factory import DetectorFactory
from .body_detector import MediaPipeBodyDetector, BodyDetector
from .hand_detector import MediaPipeHandDetector, HandDetector
from .face_detector import MediaPipeFaceDetector, FaceDetector
from .unified_detector import UnifiedDetector

# Register MediaPipe detectors in the factory
DetectorFactory.register_body_detector('mediapipe', MediaPipeBodyDetector)
DetectorFactory.register_hand_detector('mediapipe', MediaPipeHandDetector)
DetectorFactory.register_face_detector('mediapipe', MediaPipeFaceDetector)

__all__ = [
    'BaseDetector',
    'BaseBodyDetector',
    'BaseHandDetector',
    'BaseFaceDetector',
    'DetectorFactory',
    'MediaPipeBodyDetector',
    'MediaPipeHandDetector',
    'MediaPipeFaceDetector',
    'BodyDetector',
    'HandDetector',
    'FaceDetector',
    'UnifiedDetector'
]
