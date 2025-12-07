"""
Facial expression detection using MediaPipe Face Mesh.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from .base_detector import BaseFaceDetector


class MediaPipeFaceDetector(BaseFaceDetector):
    """Facial expression detection using MediaPipe Face Mesh."""
    
    def __init__(self, config: Union[Dict, None] = None):
        """
        Init face detector.
        
        Args:
            config: Configuration dictionary. If None, default values are used.
                   Expected keys:
                   - max_num_faces (int): Maximum number of faces to detect
                   - min_detection_confidence (float): Minimum detection confidence
                   - min_tracking_confidence (float): Minimum tracking confidence
                   - refine_landmarks (bool): Refine landmarks for eyes and lips
        """
        if config is None:
            config = {}
        
        # Extract parameters from config
        max_num_faces = config.get('max_num_faces', 1)
        min_detection_confidence = config.get('min_detection_confidence', 0.5)
        min_tracking_confidence = config.get('min_tracking_confidence', 0.5)
        refine_landmarks = config.get('refine_landmarks', True)
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Indices of important landmarks
        self.landmark_indices = {
            # Face contour
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            
            # Eyes
            'left_eye': [33, 160, 158, 133, 153, 144],
            'right_eye': [362, 385, 387, 263, 373, 380],
            
            # Eyebrows
            'left_eyebrow': [70, 63, 105, 66, 107],
            'right_eyebrow': [336, 296, 334, 293, 300],
            
            # Mouth outer
            'mouth_outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
            
            # Mouth inner
            'mouth_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
            
            # Nose
            'nose': [1, 2, 98, 327]
        }
    
    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detects face and facial landmarks in an image.
        
        Args:
            image: Image in BGR format (OpenCV)
            
        Returns:
            Dictionary with facial landmarks or None if no detection
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process image
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # Take the first detected face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract all landmarks (468 points)
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z  # Profundidad relativa
            })
        
        return {
            'landmarks': landmarks,
            'raw_results': results
        }
    
    def draw(self, image: np.ndarray, detection_result: Optional[Dict], 
             draw_tesselation: bool = False, draw_contours: bool = True) -> np.ndarray:
        """
        Draws facial landmarks on the image.
        
        Args:
            image: Image in BGR format
            detection_result: Result from detect()
            draw_tesselation: If True, draws full mesh (468 points)
            draw_contours: If True, draws only important contours
            
        Returns:
            Image with face mesh drawn
        """
        if detection_result is None:
            return image
        
        results = detection_result['raw_results']
        face_landmarks = results.multi_face_landmarks[0]
        
        if draw_tesselation:
            # Draw full mesh
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        
        if draw_contours:
            # Draw important contours
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        # Draw eyes and irises with refinement
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        )
        
        return image
    
    def get_expression_features(self, detection_result: Optional[Dict]) -> Optional[Dict]:
        """
        Extracts facial expression features.
        
        Args:
            detection_result: Result from detect()
            
        Returns:
            Dictionary with expression metrics
        """
        if detection_result is None:
            return None
        
        landmarks = detection_result['landmarks']
        
        def euclidean_distance(p1: Dict, p2: Dict) -> float:
            """Calculates Euclidean distance between two points."""
            return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
        
        features = {}
        
        try:
            # Mouth openness (vertical)
            upper_lip = landmarks[13]  # Upper lip top point
            lower_lip = landmarks[14]  # Lower lip bottom point
            features['mouth_openness'] = euclidean_distance(upper_lip, lower_lip)
            
            # Smile width (horizontal)
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]
            features['smile_width'] = euclidean_distance(left_mouth, right_mouth)
            
            # Eye openness (approximate)
            # Left eye
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            features['left_eye_openness'] = euclidean_distance(left_eye_top, left_eye_bottom)
            
            # Right eye
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]
            features['right_eye_openness'] = euclidean_distance(right_eye_top, right_eye_bottom)
            
            # Eyebrow raise (compared to eyes)
            left_eyebrow = landmarks[70]
            left_eye_center = landmarks[33]
            features['left_eyebrow_raise'] = abs(left_eyebrow['y'] - left_eye_center['y'])
            
            right_eyebrow = landmarks[300]
            right_eye_center = landmarks[263]
            features['right_eyebrow_raise'] = abs(right_eyebrow['y'] - right_eye_center['y'])
            
            # Detect basic expressions
            features['expression'] = self._classify_expression(features)
            
        except (IndexError, KeyError):
            return None
        
        return features
    
    def _classify_expression(self, features: Dict) -> str:
        """
        Classifies facial expression based on features.
        
        Args:
            features: Dictionary with extracted features
            
        Returns:
            Name of the detected expression
        """
        mouth_open = features['mouth_openness']
        smile_width = features['smile_width']
        left_eye = features['left_eye_openness']
        right_eye = features['right_eye_openness']
        
        # Empirical thresholds (adjust as needed)
        if mouth_open > 0.05:
            return "surprised" if (left_eye + right_eye) / 2 > 0.02 else "mouth_open"
        elif smile_width > 0.35:
            return "smiling"
        elif (left_eye + right_eye) / 2 < 0.01:
            return "eyes_closed"
        else:
            return "neutral"
    
    def get_head_pose(self, detection_result: Optional[Dict], 
                      camera_matrix: Optional[np.ndarray] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Estimates head pose (rotation and translation).
        
        Args:
            detection_result: Result from detect()
            camera_matrix: Camera intrinsic matrix (if calibrated)
            
        Returns:
            Tuple (rvec, tvec) with rotation and translation vectors
        """
        if detection_result is None:
            return None
        
        landmarks = detection_result['landmarks']
        
        # 3D model points of the face (approximate values in cm)
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -5.0, -3.0),        # Chin
            (-3.5, 5.0, -2.0),        # Left eye
            (3.5, 5.0, -2.0),         # Right eye
            (-2.5, -1.5, -2.0),       # Left mouth corner
            (2.5, -1.5, -2.0)         # Right mouth corner
        ])
        
        # Corresponding landmark indices
        indices = [1, 152, 33, 263, 61, 291]
        
        # 2D image points (normalize to pixels)
        # Assuming 640x480 image if camera_matrix is not provided
        size = (640, 480)
        
        image_points = np.array([
            (landmarks[idx]['x'] * size[0], landmarks[idx]['y'] * size[1])
            for idx in indices
        ], dtype="double")
        
        # Default camera matrix if not provided
        if camera_matrix is None:
            focal_length = size[0]
            center = (size[0] / 2, size[1] / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
        
        # Distortion coefficients (assuming no distortion)
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
        
        return rvec, tvec
    
    def close(self):
        """Releases resources."""
        self.face_mesh.close()
    
    def get_model_info(self) -> Dict:
        """Returns information about the model."""
        return {
            'backend': 'mediapipe',
            'model': 'face_mesh',
            'version': mp.__version__
        }


# Alias for compatibility with existing code
FaceDetector = MediaPipeFaceDetector
