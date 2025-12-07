"""
Factory for creating detectors based on configuration.
Allows selecting the detection backend dynamically.
"""

from typing import Dict, Any
from .base_detector import BaseBodyDetector, BaseHandDetector, BaseFaceDetector


class DetectorFactory:
    """Factory for creating detectors based on configuration."""
    
    # Registry of available detectors
    _body_detectors = {}
    _hand_detectors = {}
    _face_detectors = {}
    
    @classmethod
    def register_body_detector(cls, name: str, detector_class):
        """Registers a body pose detector."""
        cls._body_detectors[name.lower()] = detector_class
    
    @classmethod
    def register_hand_detector(cls, name: str, detector_class):
        """Registers a hand detector."""
        cls._hand_detectors[name.lower()] = detector_class
    
    @classmethod
    def register_face_detector(cls, name: str, detector_class):
        """Registers a face detector."""
        cls._face_detectors[name.lower()] = detector_class
    
    @classmethod
    def create_body_detector(cls, config: Dict[str, Any]) -> BaseBodyDetector:
        """
        Creates a body pose detector based on configuration.
        
        Args:
            config: Detector configuration
            
        Returns:
            Detector instance
        """
        backend = config.get('backend', 'mediapipe').lower()
        
        if backend not in cls._body_detectors:
            available = ', '.join(cls._body_detectors.keys())
            raise ValueError(
                f"Backend '{backend}' not available for body detector. "
                f"Available: {available}"
            )
        
        detector_class = cls._body_detectors[backend]
        return detector_class(config)
    
    @classmethod
    def create_hand_detector(cls, config: Dict[str, Any]) -> BaseHandDetector:
        """
        Creates a hand detector based on configuration.
        
        Args:
            config: Detector configuration
            
        Returns:
            Detector instance
        """
        backend = config.get('backend', 'mediapipe').lower()
        
        if backend not in cls._hand_detectors:
            available = ', '.join(cls._hand_detectors.keys())
            raise ValueError(
                f"Backend '{backend}' not available for hand detector. "
                f"Available: {available}"
            )
        
        detector_class = cls._hand_detectors[backend]
        return detector_class(config)
    
    @classmethod
    def create_face_detector(cls, config: Dict[str, Any]) -> BaseFaceDetector:
        """
        Creates a face detector based on configuration.
        
        Args:
            config: Detector configuration
            
        Returns:
            Detector instance
        """
        backend = config.get('backend', 'mediapipe').lower()
        
        if backend not in cls._face_detectors:
            available = ', '.join(cls._face_detectors.keys())
            raise ValueError(
                f"Backend '{backend}' not available for face detector. "
                f"Available: {available}"
            )
        
        detector_class = cls._face_detectors[backend]
        return detector_class(config)
    
    @classmethod
    def list_available_backends(cls) -> Dict[str, list]:
        """
        Lists all available backends.
        
        Returns:
            Dictionary with lists of backends by type
        """
        return {
            'body': list(cls._body_detectors.keys()),
            'hands': list(cls._hand_detectors.keys()),
            'face': list(cls._face_detectors.keys())
        }
