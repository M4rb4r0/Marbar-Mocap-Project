"""
BVH (Biovision Hierarchy) Exporter for MediaPipe mocap data.

BVH is a standard format for motion capture data used in:
- Blender
- Maya
- MotionBuilder
- Unreal Engine
- Unity (with plugins)
"""

import numpy as np
from typing import List, Dict, Tuple


class BVHExporter:
    """
    Export MediaPipe pose data to BVH format.
    
    BVH Structure:
    1. HIERARCHY: Defines skeleton structure with joints and offsets
    2. MOTION: Contains animation data (rotations per frame)
    """
    
    # MediaPipe Pose landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    def __init__(self, frame_time: float = 0.033333):
        """
        Initialize BVH exporter.
        
        Args:
            frame_time: Time between frames in seconds (default: ~30fps)
        """
        self.frame_time = frame_time
        self.frames = []
        self.skeleton = self._create_skeleton_hierarchy()
        
    def _create_skeleton_hierarchy(self) -> str:
        """
        Create simplified BVH skeleton hierarchy.
        Using a minimal skeleton that's easier to work with.
        
        Returns:
            BVH HIERARCHY section as string
        """
        hierarchy = """HIERARCHY
ROOT Hips
{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Chest
    {
        OFFSET 0.0 10.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Neck
        {
            OFFSET 0.0 10.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Head
            {
                OFFSET 0.0 5.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {
                    OFFSET 0.0 5.0 0.0
                }
            }
        }
        JOINT LeftShoulder
        {
            OFFSET -5.0 8.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftElbow
            {
                OFFSET -10.0 0.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT LeftWrist
                {
                    OFFSET -10.0 0.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {
                        OFFSET -3.0 0.0 0.0
                    }
                }
            }
        }
        JOINT RightShoulder
        {
            OFFSET 5.0 8.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightElbow
            {
                OFFSET 10.0 0.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT RightWrist
                {
                    OFFSET 10.0 0.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {
                        OFFSET 3.0 0.0 0.0
                    }
                }
            }
        }
    }
    JOINT LeftHip
    {
        OFFSET -5.0 0.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftKnee
        {
            OFFSET 0.0 -15.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftAnkle
            {
                OFFSET 0.0 -15.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {
                    OFFSET 0.0 -3.0 5.0
                }
            }
        }
    }
    JOINT RightHip
    {
        OFFSET 5.0 0.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightKnee
        {
            OFFSET 0.0 -15.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightAnkle
            {
                OFFSET 0.0 -15.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {
                    OFFSET 0.0 -3.0 5.0
                }
            }
        }
    }
}
"""
        return hierarchy
    
    def add_frame(self, landmarks: List[Dict[str, float]]):
        """
        Add a frame of motion capture data.
        
        Args:
            landmarks: List of 33 MediaPipe body landmarks with x, y, z coordinates
        """
        if len(landmarks) < 33:
            print(f"Warning: Expected 33 landmarks, got {len(landmarks)}")
            return
        
        # Convert landmarks to BVH frame data
        frame_data = self._landmarks_to_bvh_frame(landmarks)
        self.frames.append(frame_data)
    
    def _landmarks_to_bvh_frame(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """
        Convert MediaPipe landmarks to BVH frame data (positions + rotations).
        
        Args:
            landmarks: MediaPipe landmarks
            
        Returns:
            List of floats representing BVH frame data
        """
        # Convert all landmarks to BVH coordinate space
        def to_bvh_space(lm):
            """Convert MediaPipe normalized coords to BVH space (cm, Y-up)."""
            scale = 170.0  # Approximate human height in cm
            x = (lm['x'] - 0.5) * scale
            y = (0.5 - lm['y']) * scale  # Flip Y: MediaPipe Y-down to BVH Y-up
            z = -lm['z'] * scale * 2  # Z depth, negative for forward
            return np.array([x, y, z])
        
        # Get all joint positions
        left_hip = to_bvh_space(landmarks[self.LEFT_HIP])
        right_hip = to_bvh_space(landmarks[self.RIGHT_HIP])
        left_shoulder = to_bvh_space(landmarks[self.LEFT_SHOULDER])
        right_shoulder = to_bvh_space(landmarks[self.RIGHT_SHOULDER])
        left_elbow = to_bvh_space(landmarks[self.LEFT_ELBOW])
        right_elbow = to_bvh_space(landmarks[self.RIGHT_ELBOW])
        left_wrist = to_bvh_space(landmarks[self.LEFT_WRIST])
        right_wrist = to_bvh_space(landmarks[self.RIGHT_WRIST])
        left_knee = to_bvh_space(landmarks[self.LEFT_KNEE])
        right_knee = to_bvh_space(landmarks[self.RIGHT_KNEE])
        left_ankle = to_bvh_space(landmarks[self.LEFT_ANKLE])
        right_ankle = to_bvh_space(landmarks[self.RIGHT_ANKLE])
        nose = to_bvh_space(landmarks[self.NOSE])
        
        # Root position (hip center)
        hip_center = (left_hip + right_hip) / 2.0
        frame_data = [hip_center[0], hip_center[1], hip_center[2]]
        
        # Calculate rotations using direction vectors
        # Hips rotation (body facing direction)
        hips_rot = self._calc_rotation_from_points(left_hip, right_hip, left_shoulder)
        frame_data.extend(hips_rot)
        
        # Chest (spine direction)
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        chest_rot = self._calc_bone_rot(hip_center, shoulder_center)
        frame_data.extend(chest_rot)
        
        # Neck
        neck_rot = [0.0, 0.0, 0.0]
        frame_data.extend(neck_rot)
        
        # Head
        head_rot = self._calc_bone_rot(shoulder_center, nose)
        frame_data.extend(head_rot)
        
        # Left arm
        left_shoulder_rot = [0.0, 0.0, 0.0]
        frame_data.extend(left_shoulder_rot)
        
        left_elbow_rot = self._calc_bone_rot(left_shoulder, left_elbow)
        frame_data.extend(left_elbow_rot)
        
        left_wrist_rot = self._calc_bone_rot(left_elbow, left_wrist)
        frame_data.extend(left_wrist_rot)
        
        # Right arm
        right_shoulder_rot = [0.0, 0.0, 0.0]
        frame_data.extend(right_shoulder_rot)
        
        right_elbow_rot = self._calc_bone_rot(right_shoulder, right_elbow)
        frame_data.extend(right_elbow_rot)
        
        right_wrist_rot = self._calc_bone_rot(right_elbow, right_wrist)
        frame_data.extend(right_wrist_rot)
        
        # Left leg
        left_hip_rot = [0.0, 0.0, 0.0]
        frame_data.extend(left_hip_rot)
        
        left_knee_rot = self._calc_bone_rot(left_hip, left_knee)
        frame_data.extend(left_knee_rot)
        
        left_ankle_rot = self._calc_bone_rot(left_knee, left_ankle)
        frame_data.extend(left_ankle_rot)
        
        # Right leg
        right_hip_rot = [0.0, 0.0, 0.0]
        frame_data.extend(right_hip_rot)
        
        right_knee_rot = self._calc_bone_rot(right_hip, right_knee)
        frame_data.extend(right_knee_rot)
        
        right_ankle_rot = self._calc_bone_rot(right_knee, right_ankle)
        frame_data.extend(right_ankle_rot)
        
        return frame_data
    
    def _calc_bone_rot(self, from_pos: np.ndarray, to_pos: np.ndarray) -> List[float]:
        """
        Calculate rotation from bone direction vector (simple and robust).
        Already in BVH space (Y-up).
        """
        direction = to_pos - from_pos
        length = np.linalg.norm(direction)
        
        if length < 0.001:
            return [0.0, 0.0, 0.0]
        
        direction = direction / length
        
        # Rotation around Y axis (left/right turn)
        angle_y = np.degrees(np.arctan2(direction[0], direction[2]))
        
        # Rotation around X axis (up/down tilt)
        horizontal_dist = np.sqrt(direction[0]**2 + direction[2]**2)
        angle_x = np.degrees(np.arctan2(direction[1], horizontal_dist))
        
        # Rotation around Z axis (twist) - keep at 0
        angle_z = 0.0
        
        # Clamp angles to reasonable ranges
        angle_x = np.clip(angle_x, -180, 180)
        angle_y = np.clip(angle_y, -180, 180)
        
        return [angle_z, angle_x, angle_y]
    
    def _calc_rotation_from_points(self, left: np.ndarray, right: np.ndarray, front: np.ndarray) -> List[float]:
        """
        Calculate rotation using 3 points to define orientation plane.
        Used for torso rotation.
        """
        # Side direction (left to right)
        side = right - left
        side = side / (np.linalg.norm(side) + 1e-6)
        
        # Forward calculation
        center = (left + right) / 2.0
        to_front = front - center
        to_front = to_front / (np.linalg.norm(to_front) + 1e-6)
        
        # Project to horizontal plane for yaw
        angle_y = np.degrees(np.arctan2(to_front[0], to_front[2]))
        
        # Tilt
        angle_x = 0.0
        angle_z = 0.0
        
        return [angle_z, angle_x, angle_y]
    
    def export(self, filename: str):
        """
        Export accumulated frames to BVH file.
        
        Args:
            filename: Output filename (e.g., "capture.bvh")
        """
        if not self.frames:
            print("Warning: No frames to export")
            return
        
        with open(filename, 'w') as f:
            # Write HIERARCHY
            f.write(self.skeleton)
            f.write("\n")
            
            # Write MOTION section
            f.write("MOTION\n")
            f.write(f"Frames: {len(self.frames)}\n")
            f.write(f"Frame Time: {self.frame_time}\n")
            
            # Write frame data
            for frame in self.frames:
                frame_str = " ".join([f"{value:.6f}" for value in frame])
                f.write(f"{frame_str}\n")
        
        print(f"✓ Exported {len(self.frames)} frames to {filename}")
        print(f"  Duration: {len(self.frames) * self.frame_time:.2f} seconds")
        print(f"  FPS: {1.0 / self.frame_time:.1f}")
