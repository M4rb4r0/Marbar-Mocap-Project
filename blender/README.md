# MediaPipe Motion Capture for Blender

Real-time motion capture integration between MediaPipe and Blender Rigify rigs.

## Features

- ✅ Real-time motion capture over WebSocket
- ✅ Direct integration with Blender Rigify rigs
- ✅ 33+ body landmarks from MediaPipe
- ✅ IK-based limb animation
- ✅ Adjustable scale and mirroring
- ✅ Support for BlazePose Lite (fast) and Full models

## Installation

### 1. Install Python Dependencies (Server Side)

Already installed if you set up the main project:
```bash
pip install mediapipe opencv-python websockets
```

### 2. Install Blender Addon

#### Option A: Install websocket-client in Blender's Python
```bash
# Find Blender's Python (Windows example)
"C:\Program Files\Blender Foundation\Blender 3.6\3.6\python\bin\python.exe" -m pip install websocket-client

# Or on Linux/Mac
/path/to/blender/python/bin/python3 -m pip install websocket-client
```

#### Option B: Manual Installation (if pip doesn't work)
1. Download websocket-client from PyPI
2. Extract to Blender's site-packages:
   - Windows: `C:\Program Files\Blender Foundation\Blender 3.6\3.6\python\lib\site-packages\`
   - Linux: `/usr/share/blender/3.6/python/lib/python3.10/site-packages/`
   - Mac: `/Applications/Blender.app/Contents/Resources/3.6/python/lib/python3.10/site-packages/`

### 3. Install the Addon in Blender

1. Open Blender
2. Edit → Preferences → Add-ons
3. Click "Install..."
4. Navigate to: `blender/mediapipe_mocap_addon.py`
5. Enable the checkbox: "Animation: MediaPipe Motion Capture"
6. Save Preferences

## Usage

### Step 1: Prepare Your Rig

You need a Rigify rig named "rig" with these bones:
- `torso` (root)
- `spine_fk`, `chest`, `neck`, `head`
- `hand_ik.L`, `hand_ik.R` (IK controllers)
- `foot_ik.L`, `foot_ik.R` (IK controllers)

**Quick setup:**
1. Add → Armature → Basic → Human (Rigify)
2. Generate rig (Properties → Armature Data → Rigify → Generate Rig)
3. Name the generated rig "rig"

### Step 2: Start Python Server

In your project directory:
```bash
python main_realtime.py
```

You should see:
```
=== Realtime Streaming Started ===
WebSocket server: ws://localhost:8765
Waiting for Unity to connect...
```

### Step 3: Connect from Blender

1. In Blender 3D Viewport, press `N` to open sidebar
2. Click "MediaPipe Mocap" tab
3. Verify settings:
   - Server URL: `ws://localhost:8765`
   - Rig Name: `rig`
   - Scale: `2.0` (adjust as needed)
   - Mirror: ✓ (usually enabled)
4. Click "Connect to Server"

Your rig should now follow your movements in real-time!

## Settings

### Scale Multiplier
Adjusts the size of the captured motion:
- `1.0` = Small movements
- `2.0` = Default, natural size
- `5.0` = Exaggerated movements

### Mirror
- ✓ Enabled: Right hand in camera = right hand on rig
- ☐ Disabled: Right hand in camera = left hand on rig (mirrored)

## Troubleshooting

### "websocket-client not installed"
The addon needs the `websocket-client` Python package in Blender's Python environment.
See installation instructions above.

### Connection Failed
- Check that Python server is running
- Verify port 8765 is not blocked by firewall
- Check server URL is correct: `ws://localhost:8765`

### Rig Not Moving
- Verify rig name is exactly "rig" (case-sensitive)
- Make sure your body is visible to the camera
- Check Python console for "Body=True" messages
- Try adjusting scale multiplier

### Bones Pointing Wrong Way
- Toggle Mirror option
- Check bone names match Rigify standard
- See `BONE_MAPPING.md` for complete bone mapping

### Jittery Motion
In `config.yaml`, adjust:
```yaml
detection:
  body:
    model_complexity: 1  # Try 0 (faster) or 2 (smoother)
    min_tracking_confidence: 0.7  # Increase for stability
```

### Limbs Stretching
- Reduce Scale multiplier
- Check that camera sees full body
- Verify Z-depth values are reasonable

## Advanced Usage

### Recording Animation

Instead of real-time, you can record and import:

1. Record BVH:
```bash
python main_record_bvh.py
```

2. In Blender:
   - File → Import → Motion Capture (.bvh)
   - Select generated file from `capture/`
   - Retarget to your rig

### Custom Bone Mapping

Edit `mediapipe_mocap_addon.py` in the `apply_mocap_data()` method:

```python
# Example: Map nose to custom control
if 'my_custom_head_control' in pose_bones:
    pose_bones['my_custom_head_control'].location = landmarks[NOSE]
```

### Using Different Models

In `config.yaml`:
```yaml
detection:
  body:
    model_complexity: 0  # 0=Lite (fast), 1=Full, 2=Heavy (accurate)
```

## Bone Mapping Reference

See `BONE_MAPPING.md` for complete mapping between MediaPipe landmarks and Rigify bones.

## Performance

| Model | FPS | Quality | Use Case |
|-------|-----|---------|----------|
| Lite (0) | 40-60 | Good | Real-time preview |
| Full (1) | 20-40 | Better | General recording |
| Heavy (2) | 10-20 | Best | High-quality capture |

## Known Limitations

- Hand finger tracking requires MediaPipe Hands (work in progress)
- Facial expressions require MediaPipe Face Mesh
- Single camera = no true 3D depth (estimates from image)
- Requires clear view of full body for best results

## Next Steps

- [ ] Add hand finger retargeting
- [ ] Add facial expression capture
- [ ] Multi-camera 3D reconstruction
- [ ] Animation baking tools
- [ ] Preset rigs (Mixamo, UE5, etc.)

## Support

For issues and questions:
1. Check `BONE_MAPPING.md` for technical details
2. Verify all dependencies are installed
3. Test with `main_realtime.py` console output
4. Check Blender console for error messages (Window → Toggle System Console)
