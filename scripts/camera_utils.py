"""
Utility scripts for camera capture.
"""

import cv2
import time
import numpy as np
from typing import Optional, Tuple


class CameraCapture:
    """Class to handle camera capture."""
    
    def __init__(self, camera_id: int = 0, 
                 resolution: Tuple[int, int] = (1280, 720),
                 fps: int = 30):
        """
        Initializes the camera capture.
        
        Args:
            camera_id: Camera ID (0 for default webcam)
            resolution: Tuple (width, height) for resolution
            fps: Desired frames per second
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.target_fps = fps
        
        self.cap = None
        self.is_opened = False
        self.frame_count = 0
        self.start_time = None
    
    def open(self) -> bool:
        """
        Opens the camera.
        
        Returns:
            True if opened successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
        
        # Set resolution and FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Verify actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Cámara {self.camera_id} abierta:")
        print(f"  Resolución: {actual_width}x{actual_height} (solicitada: {self.resolution[0]}x{self.resolution[1]})")
        print(f"  FPS: {actual_fps} (solicitado: {self.target_fps})")
        
        self.is_opened = True
        self.start_time = time.time()
        self.frame_count = 0
        
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Reads a frame from the camera.
        
        Returns:
            Tuple (success, frame, timestamp)
        """
        if not self.is_opened:
            return False, None, 0.0
        
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            timestamp = time.time() - self.start_time
            return True, frame, timestamp
        
        return False, None, 0.0
    
    def get_fps(self) -> float:
        """
        Calculates the actual FPS.
        
        Returns:
            Current FPS
        """
        if self.start_time is None or self.frame_count == 0:
            return 0.0
        
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0
    
    def close(self):
        """Releases the camera."""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print(f"CCamera {self.camera_id} closed.")

def list_available_cameras(max_test: int = 5) -> list:
    """
    Lists available cameras on the system.
    
    Args:
        max_test: Maximum number of indices to test
        
    Returns:
        List of available camera indices
    """
    available = []
    
    print("Searching for available cameras...")
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  Cámara {i}: {width}x{height}")
                available.append(i)
            cap.release()
    
    if not available:
        print("  No cameras found.")
    
    return available
