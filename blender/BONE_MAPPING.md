"""
MediaPipe to Blender Rigify Bone Mapping
=========================================

This document defines the complete mapping between MediaPipe landmarks
and Blender Rigify bones for motion capture.

MediaPipe Pose Landmarks (33 total):
------------------------------------
0:  nose
1:  left_eye_inner
2:  left_eye
3:  left_eye_outer
4:  right_eye_inner
5:  right_eye
6:  right_eye_outer
7:  left_ear
8:  right_ear
9:  mouth_left
10: mouth_right
11: left_shoulder
12: right_shoulder
13: left_elbow
14: right_elbow
15: left_wrist
16: right_wrist
17: left_pinky
18: right_pinky
19: left_index
20: right_index
21: left_thumb
22: right_thumb
23: left_hip
24: right_hip
25: left_knee
26: right_knee
27: left_ankle
28: right_ankle
29: left_heel
30: right_heel
31: left_foot_index
32: right_foot_index

Additional Calculated Landmarks:
--------------------------------
33: hip_center (average of left_hip and right_hip)
34: shoulder_center (average of left_shoulder and right_shoulder)
35: pose_location (root position, same as hip_center)


Rigify Bone Mapping:
====================

TORSO & SPINE:
-------------
Bone: "torso"
- Position: hip_center (landmark 33)
- Constraint: Copy Location from hip_center
- Rotation: Based on shoulder alignment

Bone: "spine_fk.001" / "spine_fk.002" / "spine_fk.003"
- Direction: hip_center → shoulder_center
- Constraint: Copy Rotation

Bone: "chest"
- Position: shoulder_center (landmark 34)


HEAD & NECK:
-----------
Bone: "neck"
- Position: shoulder_center (landmark 34)
- Direction: shoulder_center → nose

Bone: "head"
- Position: nose (landmark 0)
- Target landmarks: eyes (1-6), ears (7-8)
- Constraint: Track To nose


LEFT ARM:
--------
Bone: "shoulder.L"
- Position: left_shoulder (landmark 11)

Bone: "upper_arm_fk.L"
- Position: left_shoulder (landmark 11)
- Direction: left_shoulder → left_elbow (13)
- Constraint: Damped Track to left_elbow

Bone: "forearm_fk.L"
- Position: left_elbow (landmark 13)
- Direction: left_elbow → left_wrist (15)
- Constraint: Damped Track to left_wrist

Bone: "hand_fk.L"
- Position: left_wrist (landmark 15)
- Rotation: Based on finger landmarks (17, 19, 21)

Bone: "hand_ik.L" (IK controller)
- Target Position: left_wrist (landmark 15)
- Pole Target: left_elbow (landmark 13)


RIGHT ARM:
---------
Bone: "shoulder.R"
- Position: right_shoulder (landmark 12)

Bone: "upper_arm_fk.R"
- Position: right_shoulder (landmark 12)
- Direction: right_shoulder → right_elbow (14)
- Constraint: Damped Track to right_elbow

Bone: "forearm_fk.R"
- Position: right_elbow (landmark 14)
- Direction: right_elbow → right_wrist (16)
- Constraint: Damped Track to right_wrist

Bone: "hand_fk.R"
- Position: right_wrist (landmark 16)
- Rotation: Based on finger landmarks (18, 20, 22)

Bone: "hand_ik.R" (IK controller)
- Target Position: right_wrist (landmark 16)
- Pole Target: right_elbow (landmark 14)


LEFT LEG:
--------
Bone: "thigh_fk.L"
- Position: left_hip (landmark 23)
- Direction: left_hip → left_knee (25)
- Constraint: Damped Track to left_knee

Bone: "shin_fk.L"
- Position: left_knee (landmark 25)
- Direction: left_knee → left_ankle (27)
- Constraint: Damped Track to left_ankle

Bone: "foot_fk.L"
- Position: left_ankle (landmark 27)
- Direction: left_ankle → left_foot_index (31)
- Additional: left_heel (29) for foot rotation

Bone: "foot_ik.L" (IK controller)
- Target Position: left_ankle (landmark 27)
- Pole Target: left_knee (landmark 25)

Bone: "toe.L"
- Position: left_foot_index (landmark 31)


RIGHT LEG:
---------
Bone: "thigh_fk.R"
- Position: right_hip (landmark 24)
- Direction: right_hip → right_knee (26)
- Constraint: Damped Track to right_knee

Bone: "shin_fk.R"
- Position: right_knee (landmark 26)
- Direction: right_knee → right_ankle (28)
- Constraint: Damped Track to right_ankle

Bone: "foot_fk.R"
- Position: right_ankle (landmark 28)
- Direction: right_ankle → right_foot_index (32)
- Additional: right_heel (30) for foot rotation

Bone: "foot_ik.R" (IK controller)
- Target Position: right_ankle (landmark 28)
- Pole Target: right_knee (landmark 26)

Bone: "toe.R"
- Position: right_foot_index (landmark 32)


CONSTRAINT CONFIGURATION:
========================

For each bone, we use Blender constraints with the following pattern:

1. COPY LOCATION (for IK targets and root)
   - Target: Empty objects positioned at landmark locations
   - Influence: 1.0
   - Mix: Replace

2. DAMPED TRACK (for bone rotations)
   - Target: Empty at child joint location
   - Track Axis: Y (Rigify default bone direction)
   - Influence: 1.0

3. IK CONSTRAINT (for limbs)
   - Target: IK controller bone
   - Pole Target: Mid-joint (elbow/knee)
   - Chain Length: 2 (upper + lower limb)


COORDINATE SYSTEM CONVERSION:
=============================

MediaPipe coordinates (normalized [0-1]):
- Origin: Top-left corner of image
- X: 0 (left) → 1 (right)
- Y: 0 (top) → 1 (bottom)
- Z: Negative (away from camera) → Positive (towards camera)

Blender World Space:
- X: Left/Right
- Y: Forward/Back
- Z: Up/Down

Conversion formula:
```python
def mediapipe_to_blender(landmark, scale=2.0, mirror=True):
    x = (landmark.x - 0.5) * scale
    y = -landmark.z * scale  # Z becomes Y (depth)
    z = (0.5 - landmark.y) * scale  # Y becomes Z (flip up/down)
    
    if mirror:
        x = -x  # Mirror horizontally
    
    return (x, y, z)
```


USAGE IN BLENDER:
=================

Method 1: Real-time with Addon (Recommended)
--------------------------------------------
1. Install addon: blender/mediapipe_mocap_addon.py
2. Create Rigify rig (or use existing humanoid)
3. Run Python server: python main_realtime.py
4. In Blender: 3D View → Sidebar → MediaPipe Mocap → Connect

Method 2: Baked Animation
-------------------------
1. Record BVH: python main_record_bvh.py
2. In Blender: File → Import → Motion Capture (.bvh)
3. Select generated .bvh file
4. Retarget to Rigify using NLA or constraints


OPTIMIZATION TIPS:
==================

1. Use IK for limbs (hands, feet) for smoother motion
2. Use FK for spine/torso for better control
3. Add damping to constraints to reduce jitter
4. Use Blender's Smooth modifier on F-curves
5. Consider using Quaternion interpolation instead of Euler
6. Add custom properties for visibility-based influence


TROUBLESHOOTING:
================

Issue: Bones pointing wrong direction
→ Check Track Axis in Damped Track constraint

Issue: Limbs stretching unnaturally
→ Reduce IK constraint influence or add stretch limits

Issue: Jittery motion
→ Increase smoothing in MediaPipe config or add Smooth modifier

Issue: Mirrored pose
→ Toggle "Mirror" option in addon or flip X coordinate

Issue: Scale too large/small
→ Adjust "Scale" parameter in addon settings
"""
