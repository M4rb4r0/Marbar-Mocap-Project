"""
Blender addon for real-time motion capture with MediaPipe.
Connects to WebSocket server and animates a Rigify rig.

Installation:
1. Open Blender
2. Edit → Preferences → Add-ons → Install
3. Select this file
4. Enable "Animation: MediaPipe Motion Capture"

Usage:
1. Create or import a Rigify rig (name it "rig")
2. Open 3D Viewport → Sidebar (N) → MediaPipe Mocap
3. Click "Connect to Server"
4. Run Python server: python main_realtime.py
"""

bl_info = {
    "name": "MediaPipe Motion Capture",
    "author": "Mocap Project",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > MediaPipe Mocap",
    "description": "Real-time motion capture using MediaPipe over WebSocket",
    "category": "Animation",
}

import bpy
import json
import mathutils
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import StringProperty, IntProperty, BoolProperty

# WebSocket client (requires websocket-client package in Blender's Python)
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("websocket-client not found. Install with: pip install websocket-client")


class MediaPipeMocapProperties(PropertyGroup):
    """Properties for MediaPipe mocap addon."""
    
    server_url: StringProperty(
        name="Server URL",
        description="WebSocket server URL",
        default="ws://localhost:8765"
    )
    
    rig_name: StringProperty(
        name="Rig Name",
        description="Name of the Rigify armature",
        default="rig"
    )
    
    is_connected: BoolProperty(
        name="Connected",
        description="Connection status",
        default=False
    )
    
    scale_multiplier: bpy.props.FloatProperty(
        name="Scale",
        description="Scale multiplier for landmarks",
        default=2.0,
        min=0.1,
        max=10.0
    )
    
    mirror_pose: BoolProperty(
        name="Mirror",
        description="Mirror the pose horizontally",
        default=True
    )


class MEDIAPIPE_OT_connect(Operator):
    """Connect to MediaPipe WebSocket server."""
    bl_idname = "mediapipe.connect"
    bl_label = "Connect to Server"
    bl_description = "Connect to the MediaPipe WebSocket server"
    
    _timer = None
    _ws = None
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            props = context.scene.mediapipe_props
            
            if not props.is_connected:
                return {'CANCELLED'}
            
            # Try to receive data (non-blocking)
            try:
                result = self._ws.recv()
                if result:
                    data = json.loads(result)
                    print(f"DEBUG: Received {len(data.get('body', []))} body landmarks")
                    self.apply_mocap_data(context, data)
            except Exception as e:
                # Non-blocking, ignore if no data
                if "timed out" not in str(e):
                    print(f"DEBUG: WebSocket error: {e}")
        
        return {'PASS_THROUGH'}
    
    def execute(self, context):
        if not WEBSOCKET_AVAILABLE:
            self.report({'ERROR'}, "websocket-client not installed")
            return {'CANCELLED'}
        
        props = context.scene.mediapipe_props
        
        try:
            # Connect to WebSocket
            self._ws = websocket.create_connection(props.server_url, timeout=2)
            props.is_connected = True
            
            # Set non-blocking
            self._ws.settimeout(0.01)
            
            # Start modal timer
            wm = context.window_manager
            self._timer = wm.event_timer_add(0.033, window=context.window)  # ~30 FPS
            wm.modal_handler_add(self)
            
            self.report({'INFO'}, f"Connected to {props.server_url}")
            return {'RUNNING_MODAL'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Connection failed: {str(e)}")
            return {'CANCELLED'}
    
    def apply_mocap_data(self, context, data):
        """Apply MediaPipe data to Rigify rig."""
        props = context.scene.mediapipe_props
        
        # Get armature
        rig = bpy.data.objects.get(props.rig_name)
        if not rig or rig.type != 'ARMATURE':
            print(f"DEBUG: Rig '{props.rig_name}' not found or not an armature")
            return
        
        body_landmarks = data.get('body', [])
        if len(body_landmarks) < 33:
            print(f"DEBUG: Not enough landmarks: {len(body_landmarks)}")
            return
        
        print(f"DEBUG: Applying mocap to rig '{props.rig_name}' with {len(body_landmarks)} landmarks")
        
        # Convert landmarks to Blender space
        def to_blender_space(lm):
            """
            Convert MediaPipe normalized coords to Blender space.
            
            MediaPipe:
            - X: 0 (left) → 1 (right)
            - Y: 0 (top) → 1 (bottom)
            - Z: negative (away) → positive (towards camera)
            
            Blender:
            - X: left/right
            - Y: forward/back (depth)
            - Z: up/down
            """
            scale = props.scale_multiplier
            
            # MediaPipe → Blender axis mapping
            x = (lm['x'] - 0.5) * scale  # Left/Right (same)
            z = (0.5 - lm['y']) * scale  # Up/Down (Y becomes Z, flipped)
            y = lm['z'] * scale * 2.0    # Depth (Z becomes Y, scaled up)
            
            if props.mirror_pose:
                x = -x
            
            return mathutils.Vector((x, y, z))
        
        # Get key landmarks
        landmarks = [to_blender_space(lm) for lm in body_landmarks]
        
        # MediaPipe indices
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
        
        # Calculate additional points
        hip_center = (landmarks[LEFT_HIP] + landmarks[RIGHT_HIP]) / 2.0
        shoulder_center = (landmarks[LEFT_SHOULDER] + landmarks[RIGHT_SHOULDER]) / 2.0
        
        # Switch to object mode first, then pose mode
        current_mode = rig.mode
        if current_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # Select the rig
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        
        # Enter pose mode
        bpy.ops.object.mode_set(mode='POSE')
        
        pose_bones = rig.pose.bones
        
        print(f"DEBUG: Available bones: {list(pose_bones.keys())[:10]}...")  # Print first 10 bones
        
        # Try different possible root bone names (Rigify, Mixamo, UE, etc.)
        root_bones = ['torso', 'root', 'hips', 'pelvis', 'hip', 'mixamorig:Hips', 'Hips']
        for bone_name in root_bones:
            if bone_name in pose_bones:
                pose_bones[bone_name].location = hip_center
                print(f"DEBUG: Set {bone_name} location to {hip_center}")
                break
        
        # Spine
        spine_bones = ['spine_fk', 'spine', 'spine.001', 'Spine', 'mixamorig:Spine', 'mixamorig:Spine1']
        for bone_name in spine_bones:
            if bone_name in pose_bones:
                spine_dir = (shoulder_center - hip_center).normalized()
                self.set_bone_direction(pose_bones[bone_name], spine_dir)
                print(f"DEBUG: Set {bone_name} direction")
                break
        
        # Head
        head_bones = ['head', 'Head', 'mixamorig:Head']
        for bone_name in head_bones:
            if bone_name in pose_bones:
                head_dir = (landmarks[NOSE] - shoulder_center).normalized()
                self.set_bone_direction(pose_bones[bone_name], head_dir)
                print(f"DEBUG: Set {bone_name} direction")
                break
        
        # Arms - Mixamo uses direct FK bones, not IK controllers
        # Left arm
        left_shoulder_bones = ['mixamorig:LeftShoulder', 'LeftShoulder', 'shoulder.L']
        left_arm_bones = ['mixamorig:LeftArm', 'LeftArm', 'upper_arm_fk.L']
        left_forearm_bones = ['mixamorig:LeftForeArm', 'LeftForeArm', 'forearm_fk.L']
        left_hand_bones = ['mixamorig:LeftHand', 'LeftHand', 'hand_fk.L', 'hand_ik.L']
        
        # Apply FK rotations for arms (Mixamo style)
        self.apply_fk_chain(pose_bones, left_arm_bones, landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW])
        self.apply_fk_chain(pose_bones, left_forearm_bones, landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST])
        
        # Right arm
        right_arm_bones = ['mixamorig:RightArm', 'RightArm', 'upper_arm_fk.R']
        right_forearm_bones = ['mixamorig:RightForeArm', 'RightForeArm', 'forearm_fk.R']
        
        self.apply_fk_chain(pose_bones, right_arm_bones, landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW])
        self.apply_fk_chain(pose_bones, right_forearm_bones, landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST])
        
        # Legs
        left_upleg_bones = ['mixamorig:LeftUpLeg', 'LeftUpLeg', 'thigh_fk.L']
        left_leg_bones = ['mixamorig:LeftLeg', 'LeftLeg', 'shin_fk.L']
        left_foot_bones = ['mixamorig:LeftFoot', 'LeftFoot', 'foot_fk.L', 'foot_ik.L']
        
        self.apply_fk_chain(pose_bones, left_upleg_bones, landmarks[LEFT_HIP], landmarks[LEFT_KNEE])
        self.apply_fk_chain(pose_bones, left_leg_bones, landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
        
        right_upleg_bones = ['mixamorig:RightUpLeg', 'RightUpLeg', 'thigh_fk.R']
        right_leg_bones = ['mixamorig:RightLeg', 'RightLeg', 'shin_fk.R']
        
        self.apply_fk_chain(pose_bones, right_upleg_bones, landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE])
        self.apply_fk_chain(pose_bones, right_leg_bones, landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
        
        # Try IK controllers if they exist (Rigify)
        self.try_ik_bones(pose_bones, 
                         ['hand_ik.L', 'hand.ik.L'], landmarks[LEFT_WRIST],
                         ['hand_ik.R', 'hand.ik.R'], landmarks[RIGHT_WRIST],
                         ['foot_ik.L', 'foot.ik.L'], landmarks[LEFT_ANKLE],
                         ['foot_ik.R', 'foot.ik.R'], landmarks[RIGHT_ANKLE])
        
        # Update view
        context.view_layer.update()
    
    def set_bone_direction(self, pose_bone, direction):
        """Set bone rotation to point in direction."""
        # Get bone's rest direction (in pose mode, bone.vector is from head to tail)
        # Most rigs have bones pointing along Y axis (0, 1, 0)
        # Mixamo rigs typically point along Y axis
        bone_vector = mathutils.Vector((0, 1, 0))  # Default bone direction (Y-axis)
        
        # For vertical bones (spine, legs), use Z-axis
        bone_name_lower = pose_bone.name.lower()
        if any(keyword in bone_name_lower for keyword in ['spine', 'neck', 'head', 'leg', 'thigh', 'shin', 'upleg']):
            bone_vector = mathutils.Vector((0, 0, 1))  # Point up for spine/legs
        
        rotation = bone_vector.rotation_difference(direction)
        pose_bone.rotation_quaternion = rotation
    
    def apply_fk_chain(self, pose_bones, bone_names, from_pos, to_pos):
        """Apply FK rotation to bone based on direction. Returns True if bone was found."""
        for bone_name in bone_names:
            if bone_name in pose_bones:
                direction = (to_pos - from_pos).normalized()
                self.set_bone_direction(pose_bones[bone_name], direction)
                print(f"DEBUG: Set FK {bone_name} direction")
                return True
        return False
    
    def try_ik_bones(self, pose_bones, left_hand_names, left_hand_pos, 
                     right_hand_names, right_hand_pos,
                     left_foot_names, left_foot_pos,
                     right_foot_names, right_foot_pos):
        """Try to apply IK if controllers exist (for Rigify rigs)."""
        for bone_name in left_hand_names:
            if bone_name in pose_bones:
                pose_bones[bone_name].location = left_hand_pos
                print(f"DEBUG: Set IK {bone_name}")
                break
        
        for bone_name in right_hand_names:
            if bone_name in pose_bones:
                pose_bones[bone_name].location = right_hand_pos
                print(f"DEBUG: Set IK {bone_name}")
                break
        
        for bone_name in left_foot_names:
            if bone_name in pose_bones:
                pose_bones[bone_name].location = left_foot_pos
                print(f"DEBUG: Set IK {bone_name}")
                break
        
        for bone_name in right_foot_names:
            if bone_name in pose_bones:
                pose_bones[bone_name].location = right_foot_pos
                print(f"DEBUG: Set IK {bone_name}")
                break
    
    def cancel(self, context):
        props = context.scene.mediapipe_props
        
        if self._timer:
            wm = context.window_manager
            wm.event_timer_remove(self._timer)
        
        if self._ws:
            self._ws.close()
        
        props.is_connected = False


class MEDIAPIPE_OT_disconnect(Operator):
    """Disconnect from MediaPipe server."""
    bl_idname = "mediapipe.disconnect"
    bl_label = "Disconnect"
    bl_description = "Disconnect from the server"
    
    def execute(self, context):
        props = context.scene.mediapipe_props
        props.is_connected = False
        self.report({'INFO'}, "Disconnected")
        return {'FINISHED'}


class MEDIAPIPE_PT_panel(Panel):
    """MediaPipe Mocap panel in 3D Viewport sidebar."""
    bl_label = "MediaPipe Mocap"
    bl_idname = "MEDIAPIPE_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MediaPipe Mocap'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.mediapipe_props
        
        # Connection settings
        box = layout.box()
        box.label(text="Connection", icon='NETWORK_DRIVE')
        box.prop(props, "server_url")
        box.prop(props, "rig_name")
        
        # Connection button
        if props.is_connected:
            box.operator("mediapipe.disconnect", icon='CANCEL')
            box.label(text="Status: Connected", icon='CHECKMARK')
        else:
            box.operator("mediapipe.connect", icon='PLAY')
            box.label(text="Status: Disconnected", icon='X')
        
        # Settings
        box = layout.box()
        box.label(text="Settings", icon='PREFERENCES')
        box.prop(props, "scale_multiplier")
        box.prop(props, "mirror_pose")
        
        # Info
        if not WEBSOCKET_AVAILABLE:
            box = layout.box()
            box.label(text="⚠ websocket-client not installed", icon='ERROR')
            box.label(text="Install with:")
            box.label(text="pip install websocket-client")


# Registration
classes = (
    MediaPipeMocapProperties,
    MEDIAPIPE_OT_connect,
    MEDIAPIPE_OT_disconnect,
    MEDIAPIPE_PT_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.mediapipe_props = bpy.props.PointerProperty(type=MediaPipeMocapProperties)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.mediapipe_props

if __name__ == "__main__":
    register()
