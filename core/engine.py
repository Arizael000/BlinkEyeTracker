"""Hilo de captura de cámara + MediaPipe + seguimiento de parpadeos.

Ejecuta el bucle pesado (lectura de cámara e inferencia) en un hilo aparte y
publica cada frame procesado junto con sus estadísticas en una cola. La interfaz
lee la cola con ``after`` para mantener Tk en el hilo principal.
"""

from __future__ import annotations

import collections
import queue
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .alarm import speak_alarm
from .tracker import (
    LEFT_EYE_POINTS,
    RIGHT_EYE_POINTS,
    BlinkTracker,
    calculate_eye_aspect_ratio,
    ratios_from_sensitivity,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "face_landmarker.task"

MAX_FACES = 5
# Resolución inicial baja para arranque rápido; se puede ajustar luego.
INITIAL_WIDTH = 640
INITIAL_HEIGHT = 480


def _build_options(model_path: Path):
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    return vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=MAX_FACES,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


class CameraWorker(threading.Thread):
    """Captura frames, detecta rostros, sigue parpadeos y publica resultados."""

    def __init__(self, settings):
        super().__init__(daemon=True)
        self.settings = settings
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._pause = threading.Event()  # Pausa el procesamiento (MediaPipe), no la cámara
        self.error: str | None = None
        self._last_frame = None
        self._last_stats = None

    # --- Control externo (llamado desde la UI) ---
    def stop(self) -> None:
        self._stop.set()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._pause.set()
        else:
            self._pause.clear()

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    # --- Bucle principal ---
    def run(self) -> None:
        if not MODEL_PATH.is_file():
            self.error = (
                f"No se encontró el modelo '{MODEL_PATH.name}' junto a la app. "
                "Colócalo en la carpeta del proyecto."
            )
            return

        # Usar CAP_DSHOW en Windows para apertura más rápida y fiable.
        cap = cv2.VideoCapture(self.settings.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            self.error = f"No se pudo abrir la cámara {self.settings.camera_index}."
            return

        # Configurar resolución baja inicial para arranque rápido.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, INITIAL_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INITIAL_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir latencia

        # Warm-up: leer unos frames para que la cámara se estabilice.
        for _ in range(3):
            cap.read()
            time.sleep(0.01)

        trackers: list[BlinkTracker] = []
        alarm_announced = False
        last_timestamp_ms = 0
        frame_times: collections.deque = collections.deque(maxlen=30)

        try:
            with vision.FaceLandmarker.create_from_options(_build_options(MODEL_PATH)) as landmarker:
                while not self._stop.is_set():
                    # SIEMPRE leer frame de la cámara para mantenerla activa y con buffer fresco.
                    ret, frame = cap.read()
                    if not ret:
                        self.error = "No se pudo leer un frame de la cámara."
                        break

                    # Aplicar brillo virtual siempre (incluso en pausa).
                    brightness = max(0, min(100, int(self.settings.brightness)))
                    if brightness > 0:
                        alpha = 1.0 + (brightness / 100.0)
                        beta = (brightness / 100.0) * 30.0
                        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

                    now = time.monotonic()

                    if self._pause.is_set():
                        # PAUSA: solo re-publicar el último frame procesado sin volver a inferir.
                        # Esto mantiene la cámara abierta y el preview vivo, pero ahorra CPU.
                        if self._last_frame is not None and self._last_stats is not None:
                            # Actualizar timestamp y FPS aparente.
                            stats = dict(self._last_stats)
                            stats["paused"] = True
                            try:
                                self.frame_queue.put_nowait((self._last_frame.copy(), stats))
                            except queue.Full:
                                pass
                        time.sleep(0.03)
                        continue

                    # --- PROCESAMIENTO NORMAL (no pausado) ---
                    height, width = frame.shape[:2]
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                    timestamp_ms = max(int(now * 1000), last_timestamp_ms + 1)
                    last_timestamp_ms = timestamp_ms

                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    faces = result.face_landmarks

                    # FPS suave (media móvil de los últimos ~30 frames).
                    frame_times.append(now)
                    fps = 0.0
                    if len(frame_times) > 1:
                        span = now - frame_times[0]
                        if span > 0:
                            fps = len(frame_times) / span

                    closed_ratio, open_ratio = ratios_from_sensitivity(self.settings.sensitivity)

                    while len(trackers) < len(faces):
                        trackers.append(BlinkTracker(timeout_seconds=self.settings.timeout_seconds))

                    face_stats = []
                    for face_index, face_landmarks in enumerate(faces):
                        tracker = trackers[face_index]
                        tracker.timeout_seconds = self.settings.timeout_seconds
                        tracker.closed_ratio = closed_ratio
                        tracker.open_ratio = open_ratio

                        left_ear = calculate_eye_aspect_ratio(face_landmarks, LEFT_EYE_POINTS)
                        right_ear = calculate_eye_aspect_ratio(face_landmarks, RIGHT_EYE_POINTS)
                        tracker.update(left_ear, right_ear, now, True)
                        warning = tracker.warning

                        if self.settings.show_landmarks:
                            for landmark in face_landmarks:
                                cv2.circle(
                                    frame,
                                    (int(landmark.x * width), int(landmark.y * height)),
                                    1,
                                    (0, 255, 0),
                                    -1,
                                )

                        status_text = (
                            f"Rostro {face_index + 1}: ALERTA | {tracker.active_elapsed:.1f}s"
                            if warning
                            else f"Rostro {face_index + 1}: Parpadeos {tracker.blink_count} | {tracker.active_elapsed:.1f}s"
                        )
                        color = (0, 0, 255) if warning else (255, 255, 255)
                        y_offset = 30 + face_index * 45
                        cv2.putText(
                            frame, status_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                        )

                        eye_status = "Cerrados" if tracker.eyes_closed else "Abiertos"
                        cv2.putText(
                            frame,
                            f"L EAR: {left_ear:.2f}  R EAR: {right_ear:.2f}  {eye_status}",
                            (10, y_offset + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1,
                        )

                        face_stats.append(
                            {
                                "index": face_index + 1,
                                "blinks": tracker.blink_count,
                                "elapsed": tracker.active_elapsed,
                                "eyes_closed": tracker.eyes_closed,
                                "left_ear": left_ear,
                                "right_ear": right_ear,
                                "warning": warning,
                            }
                        )

                    # Rostros que ya no están en pantalla: se ponen en pausa.
                    for tracker in trackers[len(faces):]:
                        tracker.update(0.0, 0.0, now, False)

                    global_warning = any(f["warning"] for f in face_stats)
                    if global_warning and not alarm_announced:
                        if self.settings.alarm_enabled:
                            speak_alarm(self.settings.alarm_message)
                        alarm_announced = True
                    elif not global_warning:
                        alarm_announced = False

                    if not faces:
                        status_text = (
                            "ALERTA: Por favor parpadea"
                            if global_warning
                            else "Rostro no detectado; temporizador pausado"
                        )
                        color = (0, 0, 255) if global_warning else (0, 165, 255)
                        cv2.putText(
                            frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
                        )

                    stats = {
                        "faces": face_stats,
                        "num_faces": len(faces),
                        "global_warning": global_warning,
                        "total_blinks": sum(f["blinks"] for f in face_stats),
                        "max_elapsed": max((f["elapsed"] for f in face_stats), default=0.0),
                        "fps": fps,
                        "paused": False,
                    }

                    # Guardar último frame/estadísticas para modo pausa.
                    self._last_frame = frame.copy()
                    self._last_stats = stats

                    try:
                        self.frame_queue.put_nowait((frame, stats))
                    except queue.Full:
                        pass
        finally:
            cap.release()