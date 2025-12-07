"""
Realtime BVH streaming Script for mocap.
Records to BVH format while streaming, saves on exit.
"""

import yaml
import cv2
import asyncio
import threading
from pose import UnifiedDetector
from scripts.camera_utils import CameraCapture
from realtime import MocapWebSocketServer
from datetime import datetime

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

    # Init WebSocket server with BVH format
    server = MocapWebSocketServer(host='localhost', port=8765, format='bvh')
    
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, args=(server,), daemon=True)
    server_thread.start()

    print(f"\n=== Realtime BVH Streaming Started ===")
    print(f"WebSocket server: ws://localhost:8765")
    print("Format: BVH (Biovision Hierarchy)")
    print("Waiting for clients to connect...")
    print("\nPress 'q' to exit and save BVH file\n")

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
                bvh_frames = len(server.bvh_exporter.frames) if hasattr(server, 'bvh_exporter') else 0
                print(f"Frame {frame_count}: Body={body_detected}, BVH frames={bvh_frames}")

            # Send data to clients via WebSocket
            if detections and server.clients:
                # Convert MediaPipe landmarks to JSON-serializable format
                body_lm = landmarks_to_list(detections.get('body'))
                
                # Handle hands (left and right)
                hands_data = detections.get('hands')
                left_hand = []
                right_hand = []
                if hands_data:
                    if hands_data.get('left') and hands_data['left'].get('landmarks'):
                        left_hand = hands_data['left']['landmarks']
                    if hands_data.get('right') and hands_data['right'].get('landmarks'):
                        right_hand = hands_data['right']['landmarks']
                
                face_lm = landmarks_to_list(detections.get('face'))
                
                data = {
                    'timestamp': timestamp,
                    'body': body_lm,
                    'left_hand': left_hand,
                    'right_hand': right_hand,
                    'face': face_lm
                }
                
                # Run async send in the server's event loop
                asyncio.run_coroutine_threadsafe(server.send_data(data), server.loop)
            
            # Show preview window
            drawn_frame = detector.draw(frame, detections)
            
            # Show recording indicator if BVH frames are being captured
            if hasattr(server, 'bvh_exporter') and server.bvh_exporter.frames:
                cv2.circle(drawn_frame, (30, 30), 10, (0, 0, 255), -1)  # Red dot
                cv2.putText(drawn_frame, f"BVH: {len(server.bvh_exporter.frames)} frames", 
                           (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow("Realtime BVH Streaming", drawn_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
    
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()
        
        # Save BVH file if frames were captured
        if hasattr(server, 'bvh_exporter') and server.bvh_exporter.frames:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture/mocap_stream_{timestamp_str}.bvh"
            
            print(f"\n=== Saving BVH file ===")
            server.bvh_exporter.export(filename)
            
            print(f"\n✓ BVH file saved: {filename}")
            print(f"  You can import this into Blender, Maya, or MotionBuilder")
        else:
            print("\n⚠ No BVH frames recorded")
        
        print("\n=== Streaming Stopped ===")
        print(f"Total frames processed: {frame_count}")
    
if __name__ == "__main__":
    main()
