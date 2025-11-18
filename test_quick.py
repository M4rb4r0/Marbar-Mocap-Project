"""
Script de prueba rápida para verificar que todo funciona.
"""

import cv2
import sys
from pose import UnifiedDetector

# Configuración básica
config = {
    'body': {
        'enabled': True,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5
    },
    'hands': {
        'enabled': True,
        'max_num_hands': 2,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5
    },
    'face': {
        'enabled': True,
        'max_num_faces': 1,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'refine_landmarks': True
    }
}

def main():
    """Prueba rápida del sistema."""
    print("\n=== Test Rápido del Sistema de Mocap ===\n")
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        print("\nIntenta:")
        print("  1. Verificar que una cámara esté conectada")
        print("  2. Ejecutar: python main.py --mode list-cameras")
        return
    
    print("Cámara abierta correctamente.")
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolución: {width}x{height}")
    
    # Inicializar detector
    print("\nInicializando detectores de MediaPipe...")
    detector = UnifiedDetector(config)
    print("Detectores listos.")
    
    print("\nPresiona 'q' para salir")
    print("Procesando frames...\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detectar
            detections = detector.detect(frame)
            
            # Dibujar
            annotated = detector.draw(frame, detections)
            
            # Info
            cv2.putText(annotated, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar
            cv2.imshow('Test - Mocap (q para salir)', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            
            # Feedback cada 30 frames
            if frame_count % 30 == 0:
                body_detected = detections['body'] is not None
                hands_detected = detections['hands'] is not None
                face_detected = detections['face'] is not None
                print(f"Frame {frame_count}: Body={body_detected}, Hands={hands_detected}, Face={face_detected}")
    
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        
        print(f"\nTest completado. Frames procesados: {frame_count}")
        print("\n✅ Si viste tu pose/manos/cara detectadas, ¡todo funciona!")
        print("\nPróximos pasos:")
        print("  python main.py --mode live      # Captura en vivo")
        print("  python main.py --mode record    # Grabar sesión")

if __name__ == "__main__":
    main()
