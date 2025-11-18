"""
Motion Capture System
Main script for pose, hand, and facial expression detection.
"""

import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from pose import UnifiedDetector
from scripts.camera_utils import CameraCapture, list_available_cameras
from scripts.data_export import DataExporter


def load_config(config_path: str = "config.yaml") -> dict:
    """Loads configuration from a YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def mode_live(config: dict):
    """Live capture mode with real-time visualization."""
    print("\n=== MODE: Live Capture ===\n")
    
    # Get configuration
    camera_id = config['cameras']['camera_ids'][0]
    resolution = tuple(config['cameras']['resolution'])
    fps = config['cameras']['fps']
    
    # Initialize camera
    camera = CameraCapture(camera_id, resolution, fps)
    if not camera.open():
        return
    
    # Inicializar detector
    detector = UnifiedDetector(config['detection'])
    
    print("\nPress 'q' to quit, 'a' for console analysis\n")
    print("Starting capture...\n")
    
    frame_count = 0
    
    try:
        while True:
            # Read frame
            ret, frame, timestamp = camera.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Detections
            detections = detector.detect(frame)
            
            # Draw detections
            annotated = detector.draw(frame, detections)
            
            # Add on-screen information
            fps_real = camera.get_fps()
            cv2.putText(annotated, f"FPS: {fps_real:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(annotated, f"Frame: {frame_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            scale = config['visualization']['preview_scale']
            if scale != 1.0:
                display_size = (int(annotated.shape[1] * scale), 
                              int(annotated.shape[0] * scale))
                annotated = cv2.resize(annotated, display_size)
            
            cv2.imshow('Mocap - Live (q para salir)', annotated)
            
            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                # Show analysis
                analysis = detector.get_full_analysis(detections)
                print(f"\n--- Frame {frame_count} ---")
                for key, value in analysis.items():
                    print(f"{key}: {value}")
            
            frame_count += 1
    
    finally:
        camera.close()
        detector.close()
        cv2.destroyAllWindows()
        print(f"\nCapture finished. Total frames: {frame_count}")


def mode_record(config: dict, output_name: str = None):
    """Session recording mode (video + landmarks)."""
    print("\n=== MODE: Record Session ===\n")
    
    # Create session name if not provided
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"Mocap_Record_{timestamp}"
    
    output_dir = Path(config['output']['output_dir']) / output_name
    
    # Camera configuration
    camera_id = config['cameras']['camera_ids'][0]
    resolution = tuple(config['cameras']['resolution'])
    fps = config['cameras']['fps']
    
    # Initialize camera
    camera = CameraCapture(camera_id, resolution, fps)
    if not camera.open():
        return
    
    # Initialize detector
    detector = UnifiedDetector(config['detection'])
    
    # Initialize exporter
    exporter = DataExporter(str(output_dir))
    exporter.set_metadata({
        'camera_id': camera_id,
        'resolution': resolution,
        'target_fps': fps,
        'config': config
    })
    
    # Initialize video writer if enabled
    if config['output']['save_raw_video']:
        exporter.init_video_writer('recording.mp4', fps, resolution)
    
    print(f"Output directory: {output_dir}")
    print("\nPress 'q' to stop recording")
    print("Press 'SPACE' to pause/resume")
    print("Recording...\n")
    
    frame_count = 0
    paused = False
    
    try:
        while True:
            # Read frame
            ret, frame, timestamp = camera.read()
            if not ret:
                print("Error reading frame")
                break
            
            if not paused:
                # Detections
                detections = detector.detect(frame)
                
                # Export landmarks
                if config['output']['save_2d_detections']:
                    landmarks = detector.export_landmarks(detections)
                    analysis = detector.get_full_analysis(detections)
                    exporter.add_frame(frame_count, timestamp, landmarks, analysis)
                
                # Draw and record video
                annotated = detector.draw(frame, detections)
                
                if config['output']['save_raw_video']:
                    exporter.write_frame(annotated)
                
                frame_count += 1
            else:
                annotated = frame.copy()
                cv2.putText(annotated, "PAUSADO", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add information
            fps_real = camera.get_fps()
            cv2.putText(annotated, f"FPS: {fps_real:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated, f"Frame: {frame_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            scale = config['visualization']['preview_scale']
            if scale != 1.0:
                display_size = (int(annotated.shape[1] * scale), 
                              int(annotated.shape[0] * scale))
                annotated_display = cv2.resize(annotated, display_size)
            else:
                annotated_display = annotated
            
            cv2.imshow('Mocap - Recording (q to quit, SPACE to pause)', annotated_display)
            
            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("PAUSADO" if paused else "REANUDADO")
    
    finally:
        camera.close()
        detector.close()
        cv2.destroyAllWindows()
        
        # Save data
        if config['output']['save_2d_detections']:
            exporter.save_json()
            exporter.save_numpy()
        
        exporter.close()
        
        print("\n" + exporter.create_summary())


def mode_process(config: dict, input_video: str):
    """Modo de procesamiento de video existente."""
    print("\n=== MODO: Procesar Video ===\n")
    
    video_path = Path(input_video)
    if not video_path.exists():
        print(f"Error: Video no encontrado: {video_path}")
        return
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path.name}")
    print(f"  Resolución: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Frames: {total_frames}\n")
    
    # Create output directory
    output_name = f"processed_{video_path.stem}"
    output_dir = Path(config['output']['output_dir']) / output_name
    
    # Initialize detector
    detector = UnifiedDetector(config['detection'])
    
    # Initialize exporter
    exporter = DataExporter(str(output_dir))
    exporter.set_metadata({
        'source_video': str(video_path),
        'resolution': (width, height),
        'fps': fps,
        'total_frames': total_frames
    })
    
    if config['output']['save_raw_video']:
        exporter.init_video_writer('processed.mp4', fps, (width, height))
    
    print(f"Procesando... (ESC para cancelar)\n")
    
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect
            detections = detector.detect(frame)
            
            # Export
            if config['output']['save_2d_detections']:
                landmarks = detector.export_landmarks(detections)
                analysis = detector.get_full_analysis(detections)
                timestamp = frame_idx / fps
                exporter.add_frame(frame_idx, timestamp, landmarks, analysis)
            
            # Draw and record
            annotated = detector.draw(frame, detections)
            
            if config['output']['save_raw_video']:
                exporter.write_frame(annotated)
            
            # Show progress
            frame_idx += 1
            if frame_idx % 30 == 0:
                progress = 100 * frame_idx / total_frames
                print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})", end='\r')
            
            # Show preview (every N frames to avoid slowing down)
            if frame_idx % 5 == 0:
                scale = 0.5
                display_size = (int(width * scale), int(height * scale))
                preview = cv2.resize(annotated, display_size)
                cv2.imshow('Processing...', preview)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    print("\nProcessing cancelled by user.")
                    break
    
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        
        # Save data
        if config['output']['save_2d_detections']:
            exporter.save_json()
            exporter.save_numpy()
        
        exporter.close()
        
        print("\n\n" + exporter.create_summary())


def mode_list_cameras():
    """List available cameras."""
    print("\n=== Listing Available Cameras ===\n")
    cameras = list_available_cameras(max_test=10)
    
    if cameras:
        print(f"\nCCameras found: {cameras}")
        print("\nUse these indices in config.yaml (camera_ids)")
    else:
        print("\nNo cameras found.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Motion Capture System with MediaPipe"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['live', 'record', 'process', 'list-cameras'],
        default='live',
        help='Operation mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input route to video (for process mode)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output session name (for record mode)'
    )
    
    args = parser.parse_args()
    
    # Special mode: list cameras (does not require config)
    if args.mode == 'list-cameras':
        mode_list_cameras()
        return
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Run selected mode
    if args.mode == 'live':
        mode_live(config)
    
    elif args.mode == 'record':
        mode_record(config, args.output)
    
    elif args.mode == 'process':
        if args.input is None:
            print("Error: Must specify --input <video> for process mode")
            return
        mode_process(config, args.input)


if __name__ == "__main__":
    main()
