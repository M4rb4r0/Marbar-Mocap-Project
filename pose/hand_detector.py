"""
Hands detector using MediaPipe Hands.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List, Union
from .base_detector import BaseHandDetector


class MediaPipeHandDetector(BaseHandDetector):
    """Hands detection using MediaPipe Hands."""
    
    def __init__(self, config: Union[Dict, None] = None):
        """
        Init hands detector.
        
        Args:
            config: Configuration dictionary. If None, default values are used.
                   Expected keys:
                   - max_num_hands (int): Number of hands to detect
                   - min_detection_confidence (float): Minimum detection confidence
                   - min_tracking_confidence (float): Minimum tracking confidence
        """
        if config is None:
            config = {}
        
        # Extract parameters from config
        max_num_hands = config.get('max_num_hands', 2)
        min_detection_confidence = config.get('min_detection_confidence', 0.5)
        min_tracking_confidence = config.get('min_tracking_confidence', 0.5)
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.landmark_names = [
            'wrist',
            'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
            'index_mcp', 'index_pip', 'index_dip', 'index_tip',
            'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
            'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
        ]
    
    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detects hands in an image.
        
        Args:
            image: Image in BGR format (OpenCV)
            
        Returns:
            Dictionary with landmarks for each hand or None if no detection
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process image
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        hands_data = {
            'left': None,
            'right': None,
            'raw_results': results
        }
        
        # Extract landmarks for each detected hand
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Determine if it is left or right hand
            handedness = results.multi_handedness[hand_idx].classification[0].label
            hand_side = 'left' if handedness == 'Left' else 'right'
            
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z 
                })
            
            hands_data[hand_side] = {
                'landmarks': landmarks,
                'handedness_confidence': results.multi_handedness[hand_idx].classification[0].score
            }
        
        return hands_data
    
    def draw(self, image: np.ndarray, detection_result: Optional[Dict]) -> np.ndarray:
        """
        Draws hand landmarks on the image.
        
        Args:
            image: Image in BGR format
            detection_result: Result from detect()
            
        Returns:
            Image with hands drawn
        """
        if detection_result is None:
            return image
        
        results = detection_result['raw_results']
        
        # Draw each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Add side labels (left/right)
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[hand_idx].classification[0].label
            confidence = results.multi_handedness[hand_idx].classification[0].score
            
            # Get wrist position for text
            h, w, _ = image.shape
            wrist = hand_landmarks.landmark[0]
            x, y = int(wrist.x * w), int(wrist.y * h)
            
            # Draw text
            cv2.putText(image, f"{handedness} ({confidence:.2f})", 
                       (x - 50, y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return image
    
    def get_gesture(self, detection_result: Optional[Dict], hand_side: str = 'right') -> Optional[str]:
        """
        Recognizes basic hand gestures.
        
        Args:
            detection_result: Result from detect()
            hand_side: 'left' or 'right'
            
        Returns:
            Name of the detected gesture or None
        """
        if detection_result is None or detection_result[hand_side] is None:
            return None
        
        landmarks = detection_result[hand_side]['landmarks']
        
        # Helper: check if a finger is extended
        def is_finger_extended(finger_tip_idx: int, finger_pip_idx: int) -> bool:
            tip_y = landmarks[finger_tip_idx]['y']
            pip_y = landmarks[finger_pip_idx]['y']
            return tip_y < pip_y  # In image coordinates, Y increases downwards
        
        # Detect extended fingers
        thumb_extended = landmarks[4]['x'] > landmarks[3]['x'] if hand_side == 'right' else landmarks[4]['x'] < landmarks[3]['x']
        index_extended = is_finger_extended(8, 6)
        middle_extended = is_finger_extended(12, 10)
        ring_extended = is_finger_extended(16, 14)
        pinky_extended = is_finger_extended(20, 18)
        
        extended_fingers = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
        
        # Recognize gestures
        if extended_fingers == 0:
            return "fist"
        elif extended_fingers == 5:
            return "open_hand"
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "peace"
        elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "pointing"
        elif thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "thumbs_up"
        
        return f"{extended_fingers}_fingers"
    
    def close(self):
        """Libera recursos."""
        self.hands.close()
    
    def get_model_info(self) -> Dict:
        """Returns information of the model."""
        return {
            'backend': 'mediapipe',
            'model': 'hands',
            'version': mp.__version__
        }


# compability alias
HandDetector = MediaPipeHandDetector
