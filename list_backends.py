"""
Script to list available detection backends.
"""

from pose import DetectorFactory


def main():
    """Lists all available backends."""
    print("\n=== Available Backends ===\n")
    
    backends = DetectorFactory.list_available_backends()
    
    print("Body Detectors:")
    for backend in backends['body']:
        print(f"  - {backend}")
    
    print("\nHand Detectors:")
    for backend in backends['hands']:
        print(f"  - {backend}")
    
    print("\nFace Detectors:")
    for backend in backends['face']:
        print(f"  - {backend}")
    
    print("\n" + "="*50)
    print("\nTo use a specific backend, edit config.yaml:")
    print("  detection:")
    print("    body:")
    print("\nAvailable Backends:")
    print("  - mediapipe: Google MediaPipe (included)")
    print("\nPlanned Backends:")
    print("  - mmpose: OpenMMLab MMPose")
    print("  - openpose: CMU OpenPose")
    print()


if __name__ == "__main__":
    main()
