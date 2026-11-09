import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configuración del modelo Face Landmarker (Asegúrate de tener 'face_landmarker.task' en la carpeta)
model_path = "face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # Modo optimizado para video en tiempo real
    num_faces=1,                            # Rastrear un solo rostro
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("--- Iniciando Detección Facial con MediaPipe Tasks ---")

# 2. Inicializar la captura de video (Cámara 0)
cap = cv2.VideoCapture(0)

# 3. Crear el detector usando un contexto
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        # Obtener dimensiones del frame para convertir coordenadas normalizadas a píxeles
        height, width, _ = frame.shape

        # 4. Preprocesamiento: Convertir BGR a RGB y empaquetarlo en mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 5. Timestamp obligatorio en milisegundos
        timestamp_ms = int(time.time() * 1000)

        # 6. Procesamiento con IA
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # 7. Dibujar los puntos clave directamente con OpenCV
        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                # Dibujar cada punto del rostro como un pequeño círculo verde
                for landmark in face_landmarks:
                    x_px = int(landmark.x * width)
                    y_px = int(landmark.y * height)
                    cv2.circle(frame, (x_px, y_px), 1, (0, 255, 0), -1)

        # 8. Mostrar el resultado
        cv2.imshow('BlinkEye - Face Landmarker Tracker', frame)

        # 9. Salir al presionar la tecla 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# 10. Liberar recursos
cap.release()
cv2.destroyAllWindows()
print("--- Detección facial finalizada. ---")