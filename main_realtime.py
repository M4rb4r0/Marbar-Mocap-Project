"""
Realtime streaming Script for mocap
"""

import yaml
import cv2
import asyncio
import threading
from pose import UnifiedDetector
from scripts.camera_utils import CameraCapture
from realtime import MocapWebSocketServer

def run_server(server):
    """Run the WebSocket server in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.start())

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Init detector
    detector = UnifiedDetector(config['detection'])

    # Init camera
    camera = CameraCapture(
        camera_id=config['cameras']['camera_ids'][0],
        resolution=tuple(config['cameras']['resolution']),
        fps=config['cameras']['fps']
    )
    camera.open()

    # Init WebSocket server
    server = MocapWebSocketServer(host='localhost', port=8765)
    
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, args=(server,), daemon=True)
    server_thread.start()

    print(f"\n=== Realtime Streaming Started ===")
    print(f"WebSocket server: ws://localhost:8765")
    print("Waiting for Unity to connect...")
    print("\nPress 'q' to exit\n")

    frame_count = 0
    
    def landmarks_to_list(detection_result):
        """Convert MediaPipe detection result to list of dicts."""
        if detection_result is None:
            return []
        # If it's already a dict with 'landmarks' key (from body_detector)
        if isinstance(detection_result, dict) and 'landmarks' in detection_result:
            return detection_result['landmarks']
        # If it has 'landmark' attribute (MediaPipe results object)
        if hasattr(detection_result, 'landmark'):
            return [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': getattr(lm, 'visibility', 1.0)} 
                    for lm in detection_result.landmark]
        return []

    try:
        while True:
            ret, frame, timestamp = camera.read()
            if not ret:
                break
            
            detections = detector.detect(frame)
            
            # Debug: verificar que MediaPipe esta detectando
            if frame_count % 30 == 0:
                body_detected = detections.get('body') is not None
                hands_detected = detections.get('hands') is not None
                print(f"Frame {frame_count}: Body={body_detected}, Hands={hands_detected}")

            # Send data to Unity/Blender via WebSocket
            if detections and server.clients:
                # Convert MediaPipe landmarks to JSON-serializable format
                body_lm = landmarks_to_list(detections.get('body'))
                
                # Calculate additional landmarks for Blender/Rigify
                additional_landmarks = {}
                if len(body_lm) >= 33:
                    # Hip center (average of left and right hip)
                    left_hip = body_lm[23]
                    right_hip = body_lm[24]
                    additional_landmarks['hip_center'] = {
                        'x': (left_hip['x'] + right_hip['x']) / 2.0,
                        'y': (left_hip['y'] + right_hip['y']) / 2.0,
                        'z': (left_hip['z'] + right_hip['z']) / 2.0,
                        'visibility': (left_hip.get('visibility', 1.0) + right_hip.get('visibility', 1.0)) / 2.0
                    }
                    
                    # Shoulder center (average of left and right shoulder)
                    left_shoulder = body_lm[11]
                    right_shoulder = body_lm[12]
                    additional_landmarks['shoulder_center'] = {
                        'x': (left_shoulder['x'] + right_shoulder['x']) / 2.0,
                        'y': (left_shoulder['y'] + right_shoulder['y']) / 2.0,
                        'z': (left_shoulder['z'] + right_shoulder['z']) / 2.0,
                        'visibility': (left_shoulder.get('visibility', 1.0) + right_shoulder.get('visibility', 1.0)) / 2.0
                    }
                    
                    # Pose location (hip center is the root position)
                    additional_landmarks['pose_location'] = additional_landmarks['hip_center'].copy()
                
                # Handle hands (left and right)
                hands_data = detections.get('hands')
                left_hand = []
                right_hand = []
                if hands_data:
                    # hands_data format: {'left': {'landmarks': [...]}, 'right': {...}}
                    if hands_data.get('left') and hands_data['left'].get('landmarks'):
                        left_hand = hands_data['left']['landmarks']
                    if hands_data.get('right') and hands_data['right'].get('landmarks'):
                        right_hand = hands_data['right']['landmarks']
                
                face_lm = landmarks_to_list(detections.get('face'))
                
                data = {
                    'timestamp': timestamp,
                    'body': body_lm,
                    'additional': additional_landmarks,
                    'left_hand': left_hand,
                    'right_hand': right_hand,
                    'face': face_lm
                }
                
                # Debug: print data once every 30 frames
                if frame_count % 30 == 0:
                    print(f"Enviando {len(body_lm)} body, {len(left_hand)} left hand, {len(right_hand)} right hand")
                
                # Run async send in the server's event loop
                asyncio.run_coroutine_threadsafe(server.send_data(data), server.loop)
            
            # Show preview window
            drawn_frame = detector.draw(frame, detections)
            cv2.imshow("Realtime Streaming", drawn_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
    
    finally:
        camera.close()
        detector.close()
        cv2.destroyAllWindows()
        print("\n=== Realtime Streaming Stopped ===")
        print(f"Total frames processed: {frame_count}")
    
if __name__ == "__main__":
    main()