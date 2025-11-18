"""
Utility scripts for exporting motion capture data to various formats.
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class DataExporter:
    """Class to export capture data to different formats."""
    
    def __init__(self, output_dir: str):
        """
        Initializes the exporter.
        
        Args:
            output_dir: Directory to save data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_data = {
            'metadata': {},
            'frames': []
        }
        
        self.video_writer = None
        self.video_path = None
    
    def set_metadata(self, metadata: Dict[str, Any]):
        """
        Sets session metadata.
        
        Args:
            metadata: Dictionary with session information
        """
        self.session_data['metadata'] = metadata
        self.session_data['metadata']['timestamp'] = datetime.now().isoformat()
    
    def add_frame(self, frame_idx: int, timestamp: float, 
                  landmarks: Dict, analysis: Optional[Dict] = None):
        """
        Adds data for a frame.
        
        Args:
            frame_idx: Frame index
            timestamp: Timestamp in seconds
            landmarks: Dictionary with detected landmarks
            analysis: Optional analysis (angles, gestures, etc.)
        """
        frame_data = {
            'frame': frame_idx,
            'timestamp': timestamp,
            'landmarks': landmarks
        }
        
        if analysis is not None:
            frame_data['analysis'] = analysis
        
        self.session_data['frames'].append(frame_data)
    
    def init_video_writer(self, filename: str, fps: int, 
                         frame_size: tuple, fourcc: str = 'mp4v'):
        """
        Initializes the video writer.
        
        Args:
            filename: Video file name
            fps: Frames per second
            frame_size: Tuple (width, height)
            fourcc: Video codec
        """
        self.video_path = self.output_dir / filename
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        
        self.video_writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc_code,
            fps,
            frame_size
        )
        
        if not self.video_writer.isOpened():
            print(f"Error: Could not create video {self.video_path}")
            self.video_writer = None
        else:
            print(f"Recording video: {self.video_path}")
    
    def write_frame(self, frame: np.ndarray):
        """
        Writes a frame to the video.
        
        Args:
            frame: Frame in BGR format
        """
        if self.video_writer is not None:
            self.video_writer.write(frame)
    
    def save_json(self, filename: str = "landmarks.json"):
        """
        Saves landmarks in JSON format.
        
        Args:
            filename: JSON file name
        """
        json_path = self.output_dir / filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, indent=2)
        
        print(f"Landmarks saved: {json_path}")
        print(f"  Total frames: {len(self.session_data['frames'])}")
    
    def save_numpy(self, filename: str = "landmarks.npz"):
        """
        Saves landmarks in compressed NumPy format.
        
        Args:
            filename: NPZ file name
        """
        npz_path = self.output_dir / filename
        
        # Convert to NumPy arrays
        arrays = {}
        
        # Metadata
        arrays['metadata'] = np.array([self.session_data['metadata']], dtype=object)
        
        # Extract landmarks by type
        for frame_data in self.session_data['frames']:
            frame_idx = frame_data['frame']
            landmarks = frame_data['landmarks']
            
            # Body landmarks
            if 'body' in landmarks and landmarks['body'] is not None:
                key = f'body_{frame_idx}'
                arrays[key] = np.array([[lm['x'], lm['y'], lm['z'], lm.get('visibility', 1.0)] 
                                       for lm in landmarks['body']])
            
            # Hand landmarks
            if 'hands' in landmarks and landmarks['hands'] is not None:
                for side in ['left', 'right']:
                    if side in landmarks['hands'] and landmarks['hands'][side] is not None:
                        key = f'hand_{side}_{frame_idx}'
                        arrays[key] = np.array([[lm['x'], lm['y'], lm['z']] 
                                               for lm in landmarks['hands'][side]])
            
            # Face landmarks
            if 'face' in landmarks and landmarks['face'] is not None:
                key = f'face_{frame_idx}'
                arrays[key] = np.array([[lm['x'], lm['y'], lm['z']] 
                                       for lm in landmarks['face']])
        
        np.savez_compressed(npz_path, **arrays)
        print(f"Landmarks saved (NumPy): {npz_path}")
    
    def close(self):
        """Closes the video writer and saves data."""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"Video saved: {self.video_path}")
            self.video_writer = None
    
    def create_summary(self) -> str:
        """
        Creates a summary of the session.
        
        Returns:
            Summary string
        """
        total_frames = len(self.session_data['frames'])
        
        if total_frames == 0:
            return "No frame data available."
        
        duration = self.session_data['frames'][-1]['timestamp'] if total_frames > 0 else 0.0
        avg_fps = total_frames / duration if duration > 0 else 0.0
        
        # Count detections
        body_count = sum(1 for f in self.session_data['frames'] 
                        if f['landmarks'].get('body') is not None)
        hands_count = sum(1 for f in self.session_data['frames'] 
                         if f['landmarks'].get('hands') is not None)
        face_count = sum(1 for f in self.session_data['frames'] 
                        if f['landmarks'].get('face') is not None)
        
        summary = f"""
Session Summary:
  Duration: {duration:.2f} seconds
  Total frames: {total_frames}
  Average FPS: {avg_fps:.2f}
  
Detections:
  Body: {body_count}/{total_frames} frames ({100*body_count/total_frames:.1f}%)
  Hands: {hands_count}/{total_frames} frames ({100*hands_count/total_frames:.1f}%)
  Face: {face_count}/{total_frames} frames ({100*face_count/total_frames:.1f}%)
  
Output directory: {self.output_dir}
"""
        return summary
