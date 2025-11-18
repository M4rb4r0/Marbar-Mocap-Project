# Proyecto Mocap - Sistema de Captura de Movimiento

Sistema de captura de movimiento (Motion Capture) con detección de:
- **Torso y brazos** (pose corporal)
- **Manos** (dedos y gestos)
- **Expresiones faciales** (ojos, boca, cejas)

Actualmente usando 1 sola webcam

## Librerias
Importante: Usar python 3.10 para compatibilidad con PyTorch y MediaPipe
- **PyTorch** + CUDA para procesamiento
- **MediaPipe** para detección de pose, manos y cara
- **OpenCV** para captura y procesamiento de video
- **NumPy/SciPy** para triangulación 3D y transformaciones
- **Matplotlib** para visualización 3D

## Estructura del Proyecto

```
Proyecto-Mocap/
├── pose/                     # Detección MediaPipe
│   ├── __init__.py
│   ├── body_detector.py      # Detección de torso/brazos (33 landmarks)
│   ├── hand_detector.py      # Detección de manos (21 landmarks x 2)
│   ├── face_detector.py      # Detección facial (468 landmarks)
│   └── unified_detector.py   # Integrador de todos los detectores
│
├── scripts/                  # Utilidades
│   ├── __init__.py
│   ├── camera_utils.py       # Captura de cámara
│   └── data_export.py        # Exportación JSON/NumPy
│
├── data/                     # Datos capturados
│   └── recordings/           # Sesiones grabadas
│
├── logs/                     # Logs del sistema
├── config.yaml               # Configuración principal
├── requirements.txt          # Dependencias
├── main.py                   # Script principal
├── test_quick.py             # Test rápido
└── README.md                 # Documentación completa
```

## ⚙️ Instalación

### 1. Crear y activar venv
```powershell
  py -3.10 -m venv venv
  .\mocap\Scripts\Activate.ps1
```

### 2. Instalar dependencias
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python mediapipe numpy scipy onnxruntime-gpu matplotlib PyYAML
```

## Uso de Cámara única

### Opción 1: Captura en tiempo real
```powershell
python main.py --mode live
```
Abre la cámara y muestra detecciones en tiempo real (pose + manos + cara).

### Opción 2: Grabar sesión
```powershell
python main.py --mode record --output data/recordings/Mocap_Record_001
```
Graba video con detecciones superpuestas y guarda landmarks en JSON.

### Opción 3: Procesar video existente
```powershell
python main.py --mode process --input video.mp4
```
Procesa un video y extrae landmarks de pose, manos y cara.

### Opción 4: Visualizar datos guardados
```powershell
python main.py --mode visualize --input data/recordings/session_001
```

## Workflow Actual (Fase 1)

1. **Configuración inicial**: Editar `config.yaml` con ID de cámara (por defecto: 0)
2. **Prueba en vivo**: Ejecutar `python main.py --mode live` para ver detecciones en tiempo real
3. **Grabar sesiones**: Usar `--mode record` para guardar video + landmarks
4. **Analiza datos**: Los landmarks 2D se guardan en JSON para análisis o integración

## Próxima Fase: Multicámara

Una vez validado el sistema con 1 cámara:
1. Agregar segunda cámara a `config.yaml`
2. Calibrar cámaras (relación espacial entre ellas)
3. Triangular puntos 2D → 3D
4. Exportar a formato 3D (BVH/FBX) para Unity/Blender

## Configuración de Cámara

Editar `config.yaml` para ajustar:
- `camera_ids`: Índice USB (ej: [0]) o URL RTSP
- `resolution`: Resolución de captura (por defecto: [1280, 720])
- `fps`: Frames por segundo (recomendado: 30)

Para encontrar cámara:
```powershell
python -c "import cv2; print([i for i in range(4) if cv2.VideoCapture(i).isOpened()])"
```

## Outputs

Los datos capturados se guardan en:
- `data/recordings/<session_name>/`
  - `camera_<id>/frames/`: Frames de cada cámara
  - `camera_<id>/detections_2d.json`: Landmarks 2D por frame
  - `reconstruction_3d.json`: Coordenadas 3D trianguladas
  - `metadata.json`: Info de la sesión

## 📊 Formato de Datos 2D

Ejemplo de estructura JSON:
```json
{
  "frame_0": {
    "timestamp": 0.0,
    "body": [[x, y, visibility], ...],  // 33 landmarks de pose (coords normalizadas 0-1)
    "hands": {
      "left": [[x, y, z], ...],   // 21 landmarks por mano (z es profundidad relativa)
      "right": [[x, y, z], ...]
    },
    "face": [[x, y, z], ...]  // 468 landmarks faciales
  }
}
```

Nota: Con una cámara obtenemos coordenadas 2D (x, y) normalizadas. MediaPipe también proporciona profundidad relativa (z).

## Troubleshooting

- **Cámaras no detectadas**: Verificar los IDs con `python -c "import cv2; print([i for i in range(4) if cv2.VideoCapture(i).isOpened()])"`
- **Baja FPS**: Reducir la resolución o desactiva detecciones no necesarias en `config.yaml`

## Roadmap

### Fase 1: Cámara única (Actual)
- [x] Configuración base del proyecto
- [x] Detección de pose, manos y cara con MediaPipe
- [x] Captura en tiempo real con overlay
- [x] Grabación de sesiones con timestamps
- [x] Exportación de landmarks 2D a JSON

### Fase 2: Multicámara
- [ ] Calibración de cámaras múltiples
- [ ] Sincronización de frames
- [ ] Triangulación 2D → 3D
- [ ] Visualización 3D en tiempo real

### Fase 3: Integración
- [ ] Exportación a BVH/FBX
- [ ] Plugin para Unity (streaming en tiempo real)
- [ ] Filtro de Kalman para suavizado
- [ ] GUI de control

## 📝 Notas

- Calibra las cámaras cada vez que se cambie su posición
- Para mejores resultados, usar iluminación uniforme y fondo de 1 solo color
