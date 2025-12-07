"""
Record mocap data and export to BVH format.
Press 'q' to stop recording and save the file.
"""

import yaml
import cv2
from pose import UnifiedDetector
from scripts.camera_utils import CameraCapture
from export import BVHExporter
from datetime import datetime

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

    # Init BVH exporter
    fps = config['cameras']['fps']
    frame_time = 1.0 / fps
    bvh_exporter = BVHExporter(frame_time=frame_time)

    print(f"\n=== BVH Recording Started ===")
    print(f"FPS: {fps}")
    print(f"Frame time: {frame_time:.6f} seconds")
    print("\nStand in front of the camera and move!")
    print("Press 'q' to stop recording and save BVH file\n")

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
            
            # Add frame to BVH if body detected
            if detections and detections.get('body'):
                body_landmarks = landmarks_to_list(detections.get('body'))
                
                if len(body_landmarks) == 33:
                    bvh_exporter.add_frame(body_landmarks)
                    
                    # Status update every 30 frames
                    if frame_count % 30 == 0:
                        duration = frame_count * frame_time
                        print(f"Recording... Frame {frame_count} ({duration:.1f}s)")
            
            # Show preview window
            drawn_frame = detector.draw(frame, detections)
            
            # Show recording indicator
            if detections and detections.get('body'):
                cv2.circle(drawn_frame, (30, 30), 10, (0, 0, 255), -1)  # Red dot
                cv2.putText(drawn_frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
            
            cv2.imshow("BVH Recording", drawn_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
    
    finally:
        camera.close()
        detector.close()
        cv2.destroyAllWindows()
        
        # Export BVH file
        if bvh_exporter.frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture/mocap_{timestamp}.bvh"
            
            print(f"\n=== Saving BVH file ===")
            bvh_exporter.export(filename)
            
            print(f"\n=== Recording Complete ===")
            print(f"Total frames captured: {len(bvh_exporter.frames)}")
            print(f"\nYou can now import '{filename}' into:")
            print("  • Blender: File → Import → Motion Capture (.bvh)")
            print("  • Maya: Import")
            print("  • MotionBuilder: Import")
        else:
            print("\n⚠ No frames recorded. Make sure your body is visible to the camera.")
    
if __name__ == "__main__":
    main()
