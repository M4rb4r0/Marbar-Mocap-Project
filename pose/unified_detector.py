"""
Unified detector for body pose, hands, and face expressions.
"""

import cv2
import numpy as np
from typing import Optional, Dict
from .detector_factory import DetectorFactory


class UnifiedDetector:
    """Unified detector that combines body pose, hands, and face expressions."""
    
    def __init__(self, config: Dict):
        """
        Initializes the unified detector.
        
        Args:
            config: Configuration dictionary with settings for each detector.
                   Each detector config should include 'enabled' and 'backend' keys.
        """
        self.config = config
        
        # Initialize body detector
        body_cfg = config.get('body', {})
        self.body_detector = None
        if body_cfg.get('enabled', True):
            self.body_detector = DetectorFactory.create_body_detector(body_cfg)
        
        # Initialize hand detector
        hands_cfg = config.get('hands', {})
        self.hand_detector = None
        if hands_cfg.get('enabled', True):
            self.hand_detector = DetectorFactory.create_hand_detector(hands_cfg)
        
        # Initialize face detector
        face_cfg = config.get('face', {})
        self.face_detector = None
        if face_cfg.get('enabled', True):
            self.face_detector = DetectorFactory.create_face_detector(face_cfg)
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detects pose, hands, and face in an image.
        
        Args:
            image: Image in BGR format (OpenCV)
            
        Returns:
            Dictionary with all detections
        """
        results = {
            'body': None,
            'hands': None,
            'face': None
        }
        
        # Detect body pose
        if self.body_detector is not None:
            results['body'] = self.body_detector.detect(image)
        
        # Detect hands
        if self.hand_detector is not None:
            results['hands'] = self.hand_detector.detect(image)
        
        # Detect face
        if self.face_detector is not None:
            results['face'] = self.face_detector.detect(image)
        
        return results
    
    def draw(self, image: np.ndarray, detection_results: Dict) -> np.ndarray:
        """
        Draws all detections on the image.
        
        Args:
            image: Image in BGR format
            detection_results: Result from detect()
            
        Returns:
            Image with all detections drawn
        """
        annotated_image = image.copy()
        
        # Draw body pose
        if self.body_detector is not None and detection_results['body'] is not None:
            annotated_image = self.body_detector.draw(annotated_image, detection_results['body'])
        
        # Draw hands
        if self.hand_detector is not None and detection_results['hands'] is not None:
            annotated_image = self.hand_detector.draw(annotated_image, detection_results['hands'])
        
        # Draw face
        if self.face_detector is not None and detection_results['face'] is not None:
            annotated_image = self.face_detector.draw(
                annotated_image, 
                detection_results['face'],
                draw_tesselation=True,
                draw_contours=True
            )
        
        return annotated_image
    
    def get_full_analysis(self, detection_results: Dict) -> Dict:
        """
        Obtains full analysis of the detected pose.
        
        Args:
            detection_results: Result from detect()
            
        Returns:
            Dictionary with detailed analysis
        """
        analysis = {}
        
        # Body analysis
        if self.body_detector is not None and detection_results['body'] is not None:
            analysis['body_angles'] = self.body_detector.get_angles(detection_results['body'])
        
        # Hand analysis
        if self.hand_detector is not None and detection_results['hands'] is not None:
            analysis['left_hand_gesture'] = self.hand_detector.get_gesture(
                detection_results['hands'], 'left'
            )
            analysis['right_hand_gesture'] = self.hand_detector.get_gesture(
                detection_results['hands'], 'right'
            )
        
        # Face analysis
        if self.face_detector is not None and detection_results['face'] is not None:
            analysis['face_expression'] = self.face_detector.get_expression_features(
                detection_results['face']
            )
        
        return analysis
    
    def export_landmarks(self, detection_results: Dict) -> Dict:
        """
        Exports landmarks in a serializable (JSON) format.
        
        Args:
            detection_results: Result from detect()
            
        Returns:
            Dictionary with landmarks ready for serialization
        """
        export_data = {}
        
        # Export body pose
        if detection_results['body'] is not None:
            export_data['body'] = detection_results['body']['landmarks']
        
        # Export hands
        if detection_results['hands'] is not None:
            export_data['hands'] = {}
            if detection_results['hands']['left'] is not None:
                export_data['hands']['left'] = detection_results['hands']['left']['landmarks']
            if detection_results['hands']['right'] is not None:
                export_data['hands']['right'] = detection_results['hands']['right']['landmarks']
        
        # Export face
        if detection_results['face'] is not None:
            export_data['face'] = detection_results['face']['landmarks']
        
        return export_data
    
    def close(self):
        """Releases resources of all detectors."""
        if self.body_detector is not None:
            self.body_detector.close()
        if self.hand_detector is not None:
            self.hand_detector.close()
        if self.face_detector is not None:
            self.face_detector.close()
