"""
Base abstract class for detectors.
Defines the standard interface that all detectors must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any


class BaseDetector(ABC):
    """Base abstract class for all detectors."""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detects features in an image.
        
        Args:
            image: Image in BGR format (OpenCV)
            
        Returns:
            Dictionary with landmarks or None if no detection
            Expected format: {'landmarks': [...], 'raw_results': ...}
        """
        pass
    
    @abstractmethod
    def draw(self, image: np.ndarray, detection_result: Optional[Dict]) -> np.ndarray:
        """
        Draws detections on the image.
        
        Args:
            image: Image in BGR format
            detection_result: Result from detect()
            
        Returns:
            Image with detections drawn
        """
        pass
    
    @abstractmethod
    def close(self):
        """Releases resources of the detector."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns information about the model used.
        
        Returns:
            Dictionary with information about the model
        """
        return {
            'backend': self.__class__.__name__,
            'version': 'unknown'
        }


class BaseBodyDetector(BaseDetector):
    """Base class for body pose detectors."""
    
    @abstractmethod
    def get_angles(self, detection_result: Optional[Dict]) -> Optional[Dict[str, float]]:
        """
        Calculates important angles (elbows, knees, shoulders).
        
        Args:
            detection_result: Result from detect()
            
        Returns:
            Dictionary with angles in degrees
        """
        pass


class BaseHandDetector(BaseDetector):
    """Base class for hand detectors."""
    
    @abstractmethod
    def get_gesture(self, detection_result: Optional[Dict], hand_side: str = 'right') -> Optional[str]:
        """
        Recognizes hand gestures.
        
        Args:
            detection_result: Results from detect()
            hand_side: 'left' or 'right'
            
        Returns:
            Name of the detected gesture or None
        """
        pass


class BaseFaceDetector(BaseDetector):
    """Base class for face detectors."""
    
    @abstractmethod
    def get_expression_features(self, detection_result: Optional[Dict]) -> Optional[Dict]:
        """
        Extracts facial expression features.
        
        Args:
            detection_result: Result from detect()
            
        Returns:
            Dictionary with expression metrics
        """
        pass
