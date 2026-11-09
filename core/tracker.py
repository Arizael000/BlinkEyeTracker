"""Seguimiento de parpadeos por rostro (lógica reutilizada de parpadeo_multiple.py)."""

from __future__ import annotations

import numpy as np

# Umbrales relativos que se adaptan a cada rostro.
BLINK_CLOSED_RATIO = 0.78
BLINK_OPEN_RATIO = 0.90
EAR_BASELINE_ALPHA = 0.05

# Duraciones aceptables de un parpadeo (en segundos) para filtrar ruido.
BLINK_MIN_DURATION_SECONDS = 0.03
BLINK_MAX_DURATION_SECONDS = 0.8

# Índices MediaPipe en el orden esperado por calculate_eye_aspect_ratio.
LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]


def euclidean_distance(p1, p2) -> float:
    """Distancia euclidiana entre dos landmarks normalizados (atributos x, y)."""
    return float(np.hypot(p1.x - p2.x, p1.y - p2.y))


def calculate_eye_aspect_ratio(landmarks, eye_points) -> float:
    """
    EAR = (distancia vertical 1 + distancia vertical 2) / (2 * distancia horizontal).

    Los puntos deben estar ordenados como: esquina, párpado superior externo,
    párpado superior interno, esquina opuesta, párpado inferior interno,
    párpado inferior externo.
    """
    vertical_1 = euclidean_distance(landmarks[eye_points[1]], landmarks[eye_points[5]])
    vertical_2 = euclidean_distance(landmarks[eye_points[2]], landmarks[eye_points[4]])
    horizontal = euclidean_distance(landmarks[eye_points[0]], landmarks[eye_points[3]])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def ratios_from_sensitivity(sensitivity: float) -> tuple[float, float]:
    """Convierte sensibilidad (0-100) a (umbral de cierre, umbral de apertura).

    A mayor sensibilidad, el umbral de cierre se acerca a 1.0, de modo que
    basta un cierre parcial del párpado para contar como parpadeo.
    """
    sensitivity = max(0.0, min(100.0, float(sensitivity)))
    closed = 0.60 + 0.32 * (sensitivity / 100.0)
    opened = min(closed + 0.12, 0.98)
    return closed, opened


class BlinkTracker:
    """Mantiene el estado de parpadeo de un único rostro a lo largo del tiempo."""

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        closed_ratio: float = BLINK_CLOSED_RATIO,
        open_ratio: float = BLINK_OPEN_RATIO,
    ):
        self.timeout_seconds = timeout_seconds
        self.closed_ratio = closed_ratio
        self.open_ratio = open_ratio
        self.blink_count = 0
        self.active_elapsed = 0.0
        self.eyes_closed = False
        self.closed_since = None
        self.tracking_open_eyes = False
        self.last_update_time = None
        self.left_ear_baseline = 0.25
        self.right_ear_baseline = 0.25
        self.left_ear = 0.0
        self.right_ear = 0.0

    def update(self, left_ear: float, right_ear: float, now: float, has_face: bool) -> bool:
        """Actualiza el estado y devuelve True si se detectó un parpadeo válido."""
        blink_detected = False

        if self.last_update_time is not None and self.tracking_open_eyes:
            self.active_elapsed += max(0.0, now - self.last_update_time)
        self.last_update_time = now
        self.left_ear = left_ear
        self.right_ear = right_ear

        if not has_face:
            self.eyes_closed = False
            self.closed_since = None
            self.tracking_open_eyes = False
            return blink_detected

        left_ratio = left_ear / max(self.left_ear_baseline, 0.01)
        right_ratio = right_ear / max(self.right_ear_baseline, 0.01)
        average_ratio = (left_ratio + right_ratio) / 2.0

        if self.eyes_closed:
            if average_ratio >= self.open_ratio:
                closed_duration = now - self.closed_since
                if BLINK_MIN_DURATION_SECONDS <= closed_duration <= BLINK_MAX_DURATION_SECONDS:
                    self.blink_count += 1
                    # Reinicia el contador de "sin parpadear" al abrir los ojos tras un parpadeo válido
                    self.active_elapsed = 0.0
                    blink_detected = True
                elif closed_duration > BLINK_MAX_DURATION_SECONDS:
                    # Cierre muy largo (no es un parpadeo): también reinicia el contador
                    self.active_elapsed = 0.0
                self.eyes_closed = False
                self.closed_since = None
        elif average_ratio <= self.closed_ratio:
            # Ojos se cierran: reinicia el contador de "tiempo sin parpadear" a 0
            self.eyes_closed = True
            self.closed_since = now
            self.active_elapsed = 0.0
        elif average_ratio > self.closed_ratio and left_ear > 0.05 and right_ear > 0.05:
            # Adapta la línea base de "ojos abiertos" a las condiciones de cada rostro.
            self.left_ear_baseline += EAR_BASELINE_ALPHA * (left_ear - self.left_ear_baseline)
            self.right_ear_baseline += EAR_BASELINE_ALPHA * (right_ear - self.right_ear_baseline)

        self.tracking_open_eyes = not self.eyes_closed
        return blink_detected

    @property
    def warning(self) -> bool:
        """True cuando el rostro lleva demasiado tiempo sin parpadear."""
        return self.active_elapsed >= self.timeout_seconds
