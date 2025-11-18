#Corporal Pose Detection Module using MediaPipe Pose.
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List, Tuple


class BodyDetector:
#Corporal Pose Detection Module using MediaPipe Pose.
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1):
        """
        Init pose detector.
        
        Args:
            min_detection_confidence: Minimum confidence for initial detection
            min_tracking_confidence: Minimum confidence for tracking
            model_complexity: 0 (lite), 1 (full), 2 (heavy) - more accurate but slower
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
    
    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detects corporal pose in an image.
        
        Args:
            image: Image in BGR format (OpenCV)
            
        Returns:
            Dictionary with landmarks or None if no detection
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process image
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        return {
            'landmarks': landmarks,
            'raw_results': results
        }
    
    def draw(self, image: np.ndarray, detection_result: Optional[Dict]) -> np.ndarray:
        """
        Draws the skeleton on the image.
        
        Args:
            image: Image in BGR format
            detection_result: Result from detect()
            
        Returns:
            Image with skeleton drawn
        """
        if detection_result is None:
            return image
        
        results = detection_result['raw_results']
        
        # Dibujar landmarks y conexiones
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        return image
    
    def get_angles(self, detection_result: Optional[Dict]) -> Optional[Dict[str, float]]:
        """
        Calculates important angles (elbows, knees, shoulders).
        
        Args:
            detection_result: Result from detect()
            
        Returns:
            Dictionary with angles in degrees
        """
        if detection_result is None:
            return None
        
        landmarks = detection_result['landmarks']
        
        def calculate_angle(p1: Dict, p2: Dict, p3: Dict) -> float:
            """Calculates angle between three points."""
            a = np.array([p1['x'], p1['y']])
            b = np.array([p2['x'], p2['y']])
            c = np.array([p3['x'], p3['y']])
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            
            return np.degrees(angle)
        
        angles = {}
        
        try:
            # Left elbow (shoulder-elbow-wrist)
            angles['left_elbow'] = calculate_angle(
                landmarks[11], landmarks[13], landmarks[15]  # shoulder-elbow-wrist
            )
            
            # Right elbow
            angles['right_elbow'] = calculate_angle(
                landmarks[12], landmarks[14], landmarks[16]
            )
            
            # Left knee (hip-knee-ankle)
            angles['left_knee'] = calculate_angle(
                landmarks[23], landmarks[25], landmarks[27]  # hip-knee-ankle
            )
            
            # Right knee
            angles['right_knee'] = calculate_angle(
                landmarks[24], landmarks[26], landmarks[28]
            )
            
            # Left shoulder
            angles['left_shoulder'] = calculate_angle(
                landmarks[23], landmarks[11], landmarks[13]  # hip-shoulder-elbow
            )
            
            # Right shoulder
            angles['right_shoulder'] = calculate_angle(
                landmarks[24], landmarks[12], landmarks[14]
            )
            
        except (IndexError, ZeroDivisionError):
            return None
        
        return angles
    
    def close(self):
        """Libera recursos."""
        self.pose.close()
