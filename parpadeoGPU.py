from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
import subprocess
import sys
import time
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURACIÓN ---
# El modelo se busca junto al script, no según la carpeta desde la que se ejecute.
model_path = Path(__file__).resolve().with_name("face_landmarker.task")
base_options = python.BaseOptions(model_asset_path=str(model_path))
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Parámetros del seguimiento de ojos
# Umbrales con histéresis para evitar cambios de estado por ruido del seguimiento.
# Umbrales relativos al EAR abierto de cada ojo para compensar diferencias
# entre ojos, cámara y distancia.
EAR_CLOSED_RATIO = 0.72
EAR_OPEN_RATIO = 0.82
TIMEOUT_THRESHOLD_SECONDS = 5.0
BLINK_MIN_DURATION_SECONDS = 0.04
BLINK_MAX_DURATION_SECONDS = 0.8

print("--- Iniciando Detección de Parpadeos y Seguimiento de Tiempo ---")

# ------------------------------------------------------------
# 1. FUNCIONES AUXILIARES: Geometría
# ------------------------------------------------------------

def euclidean_distance(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos normalizados (x, y)."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_eye_aspect_ratio(landmarks, eye_points):
    """
    Los puntos deben estar ordenados como: esquina, párpado superior externo,
    párpado superior interno, esquina opuesta, párpado inferior interno,
    párpado inferior externo.
    EAR = (distancia vertical 1 + distancia vertical 2) / (2 * distancia horizontal)
    """
    vertical_dist1 = euclidean_distance(landmarks[eye_points[1]], landmarks[eye_points[5]])
    vertical_dist2 = euclidean_distance(landmarks[eye_points[2]], landmarks[eye_points[4]])
    horizontal_dist = euclidean_distance(landmarks[eye_points[0]], landmarks[eye_points[3]])

    if horizontal_dist == 0:
        return 0.0

    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

# ------------------------------------------------------------
# 2. LÓGICA DE PARSEO DE LANDMARKS
# ------------------------------------------------------------

# Índices MediaPipe en el orden esperado por calculate_eye_aspect_ratio.
LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]


class BlinkTracker:
    def __init__(self):
        self.blink_count = 0
        self.active_elapsed = 0.0
        self.eyes_closed = False
        self.closed_since = None
        self.tracking_open_eyes = False
        self.last_update_time = None
        self.left_open_baseline = None
        self.right_open_baseline = None

    def update(self, left_ear, right_ear, now, has_face):
        blink_detected = False
        if self.last_update_time is not None and self.tracking_open_eyes:
            self.active_elapsed += max(0.0, now - self.last_update_time)
        self.last_update_time = now

        if not has_face:
            self.eyes_closed = False
            self.closed_since = None
            self.tracking_open_eyes = False
            return blink_detected

        if self.left_open_baseline is None or self.right_open_baseline is None:
            self.left_open_baseline = left_ear
            self.right_open_baseline = right_ear

        left_closed_threshold = self.left_open_baseline * EAR_CLOSED_RATIO
        right_closed_threshold = self.right_open_baseline * EAR_CLOSED_RATIO
        left_open_threshold = self.left_open_baseline * EAR_OPEN_RATIO
        right_open_threshold = self.right_open_baseline * EAR_OPEN_RATIO

        if self.eyes_closed:
            eyes_reopened = (
                left_ear >= left_open_threshold
                and right_ear >= right_open_threshold
            )
            if eyes_reopened:
                closed_duration = now - self.closed_since
                if BLINK_MIN_DURATION_SECONDS <= closed_duration <= BLINK_MAX_DURATION_SECONDS:
                    self.blink_count += 1
                    self.active_elapsed = 0.0
                    blink_detected = True
                self.eyes_closed = False
                self.closed_since = None
        elif (
            left_ear <= left_closed_threshold
            and right_ear <= right_closed_threshold
        ):
            self.eyes_closed = True
            self.closed_since = now

        if not self.eyes_closed:
            self.left_open_baseline = (
                0.98 * self.left_open_baseline + 0.02 * left_ear
            )
            self.right_open_baseline = (
                0.98 * self.right_open_baseline + 0.02 * right_ear
            )

        self.tracking_open_eyes = not self.eyes_closed
        return blink_detected


def speak_alarm():
    """Usa la voz SAPI integrada en Windows sin bloquear el bucle de vídeo."""
    powershell_command = (
        "$ErrorActionPreference = 'Stop'; "
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "try { "
        "$voice = $speaker.GetInstalledVoices() | "
        "Where-Object { $_.VoiceInfo.Culture.Name -like 'es-*' } | "
        "Select-Object -First 1; "
        "if ($voice) { $speaker.SelectVoice($voice.VoiceInfo.Name) }; "
        "$speaker.Speak('Parpadea, parpadea, parpadea') "
        "} "
        "finally { $speaker.Dispose() }"
    )
    try:
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", powershell_command],
            check=True,
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.CalledProcessError) as error:
        details = getattr(error, "stderr", None)
        print(f"No se pudo reproducir la alarma hablada: {details or error}", file=sys.stderr)


# ------------------------------------------------------------
# 3. BUCLE PRINCIPAL
# ------------------------------------------------------------

# Inicialización
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara 0.")

tracker = BlinkTracker()
alarm_announced = False
last_timestamp_ms = 0

try:
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("No se pudo leer un frame de la cámara.")

            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            now = time.monotonic()
            timestamp_ms = max(int(now * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            has_face = bool(result.face_landmarks)
            left_ear = right_ear = 0.0

            if has_face:
                face_landmarks = result.face_landmarks[0]
                left_ear = calculate_eye_aspect_ratio(face_landmarks, LEFT_EYE_POINTS)
                right_ear = calculate_eye_aspect_ratio(face_landmarks, RIGHT_EYE_POINTS)

            blink_detected = tracker.update(left_ear, right_ear, now, has_face)
            warning = tracker.active_elapsed >= TIMEOUT_THRESHOLD_SECONDS

            if warning and not alarm_announced:
                threading.Thread(target=speak_alarm, daemon=True).start()
                alarm_announced = True
            elif blink_detected:
                alarm_announced = False

            if warning:
                status_text = (
                    f"ALERTA: Por favor parpadea | "
                    f"{tracker.active_elapsed:.1f}s"
                )
                color = (0, 0, 255)
            elif not has_face:
                status_text = "Rostro no detectado; temporizador pausado"
                color = (0, 165, 255)
            else:
                status_text = (
                    f"Parpadeos: {tracker.blink_count} | "
                    f"Tiempo: {tracker.active_elapsed:.1f}s"
                )
                color = (255, 255, 255)

            if has_face:
                for landmark in result.face_landmarks[0]:
                    x_px = int(landmark.x * width)
                    y_px = int(landmark.y * height)
                    cv2.circle(frame, (x_px, y_px), 1, (0, 255, 0), -1)

                if tracker.eyes_closed:
                    eye_status = "Cerrados; tiempo pausado"
                else:
                    eye_status = (
                        f"Abiertos (cierre L/R "
                        f"{tracker.left_open_baseline * EAR_CLOSED_RATIO:.2f}/"
                        f"{tracker.right_open_baseline * EAR_CLOSED_RATIO:.2f})"
                    )
                cv2.putText(
                    frame,
                    f"L EAR: {left_ear:.2f}  R EAR: {right_ear:.2f}  {eye_status}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
            cv2.imshow("BlinkEye Tracker", frame)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()

print("--- Detección facial finalizada. ---")
