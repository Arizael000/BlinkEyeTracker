from pathlib import Path
import argparse
import math
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
# Modelos: Face Mesh ONNX (Apache-2.0) y detector YuNet (MIT).
# Descargas: https://huggingface.co/senty-au/face_landmarks_detector-ONNX/resolve/337d58218b5b1cc597ca3c67360880b920f6ce7b/onnx/model.onnx
# https://media.githubusercontent.com/media/opencv/opencv_zoo/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
LANDMARK_MODEL = ROOT / "face_landmarks_478.onnx"
FACE_DETECTOR_MODEL = ROOT / "face_detection_yunet_2023mar.onnx"

# On this PC DXGI adapter 1 is AMD Radeon(TM) Graphics; adapter 0 is NVIDIA.
AMD_DML_DEVICE_ID = 1
LANDMARK_INPUT_SIZE = 256
FACE_PRESENCE_THRESHOLD = 0.5
FACE_DETECTION_THRESHOLD = 0.75
EAR_CLOSED_THRESHOLD = 0.18
EAR_OPEN_THRESHOLD = 0.22
TIMEOUT_SECONDS = 5.0
BLINK_MIN_DURATION_SECONDS = 0.04
BLINK_MAX_DURATION_SECONDS = 0.8

# Six contour points on each eye in the 478-point MediaPipe face mesh.
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

    def update(self, left_ear, right_ear, now, has_face):
        blink_detected = False
        if (
            has_face
            and self.last_update_time is not None
            and self.tracking_open_eyes
        ):
            self.active_elapsed += max(0.0, now - self.last_update_time)
        self.last_update_time = now

        if not has_face:
            self.eyes_closed = False
            self.closed_since = None
            self.tracking_open_eyes = False
            return blink_detected

        if self.eyes_closed:
            if left_ear >= EAR_OPEN_THRESHOLD and right_ear >= EAR_OPEN_THRESHOLD:
                closed_duration = now - self.closed_since
                if BLINK_MIN_DURATION_SECONDS <= closed_duration <= BLINK_MAX_DURATION_SECONDS:
                    self.blink_count += 1
                    self.active_elapsed = 0.0
                    blink_detected = True
                self.eyes_closed = False
                self.closed_since = None
        elif (
            left_ear <= EAR_CLOSED_THRESHOLD
            and right_ear <= EAR_CLOSED_THRESHOLD
        ):
            self.eyes_closed = True
            self.closed_since = now

        self.tracking_open_eyes = not self.eyes_closed
        return blink_detected


def calculate_eye_aspect_ratio(landmarks, eye_points):
    def distance(index_a, index_b):
        return math.hypot(
            float(landmarks[index_a][0] - landmarks[index_b][0]),
            float(landmarks[index_a][1] - landmarks[index_b][1]),
        )

    horizontal = distance(eye_points[0], eye_points[3])
    if horizontal <= 1e-6:
        return 0.0

    vertical_1 = distance(eye_points[1], eye_points[5])
    vertical_2 = distance(eye_points[2], eye_points[4])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def create_landmark_session(device_id):
    session_options = ort.SessionOptions()
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.enable_mem_pattern = False
    session = ort.InferenceSession(
        str(LANDMARK_MODEL),
        sess_options=session_options,
        providers=[
            ("DmlExecutionProvider", {"device_id": str(device_id)}),
            "CPUExecutionProvider",
        ],
    )
    providers = session.get_providers()
    if not providers or providers[0] != "DmlExecutionProvider":
        raise RuntimeError(
            f"ONNX Runtime no activó DirectML (providers: {providers}). "
            "Actualiza el driver AMD y comprueba la instalación de DirectML."
        )
    return session


def crop_face_square(frame, face_box, scale=1.5):
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = (float(value) for value in face_box[:4])
    size = max(1, int(math.ceil(max(width, height) * scale)))
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    left = int(round(center_x - size / 2.0))
    top = int(round(center_y - size / 2.0))

    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(frame_width, left + size), min(frame_height, top + size)
    if x1 >= x2 or y1 >= y2:
        return None, left, top, size

    crop = frame[y1:y2, x1:x2]
    crop = cv2.copyMakeBorder(
        crop,
        max(0, -top),
        max(0, top + size - frame_height),
        max(0, -left),
        max(0, left + size - frame_width),
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return crop, left, top, size


def infer_landmarks(session, input_name, frame, face_box, output_names):
    crop, left, top, size = crop_face_square(frame, face_box)
    if crop is None:
        return None

    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb_crop,
        (LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    outputs = session.run(output_names, {input_name: tensor})

    presence_logit = float(outputs[1].reshape(-1)[0])
    presence = 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, presence_logit))))
    if presence < FACE_PRESENCE_THRESHOLD:
        return None

    points = outputs[0].reshape(478, 3)[:, :2].astype(np.float32)
    points *= size / LANDMARK_INPUT_SIZE
    points[:, 0] += left
    points[:, 1] += top
    return points


def speak_alarm():
    command = (
        "$ErrorActionPreference = 'Stop'; "
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "try { "
        "$voice = $speaker.GetInstalledVoices() | "
        "Where-Object { $_.VoiceInfo.Culture.Name -like 'es-*' } | "
        "Select-Object -First 1; "
        "if ($voice) { $speaker.SelectVoice($voice.VoiceInfo.Name) }; "
        "$speaker.Speak('Por favor, parpadea') "
        "} finally { $speaker.Dispose() }"
    )
    try:
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            check=True,
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.CalledProcessError) as error:
        details = getattr(error, "stderr", None)
        print(f"No se pudo reproducir la alarma hablada: {details or error}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Detector de parpadeos con landmarks en la GPU AMD vía DirectML."
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=AMD_DML_DEVICE_ID,
        help="Índice de adaptador DirectML (en este PC, 1 = AMD Radeon integrada).",
    )
    args = parser.parse_args()

    missing_models = [
        path.name for path in (LANDMARK_MODEL, FACE_DETECTOR_MODEL) if not path.is_file()
    ]
    if missing_models:
        raise FileNotFoundError(
            "Faltan modelos: "
            + ", ".join(missing_models)
            + ". Consulta las URL indicadas en los comentarios de este script."
        )

    available_providers = ort.get_available_providers()
    if "DmlExecutionProvider" not in available_providers:
        raise RuntimeError(
            f"DirectML no está disponible en ONNX Runtime: {available_providers}. "
            "Instala requirements-gpu.txt en el intérprete activo."
        )

    landmark_session = create_landmark_session(args.device_id)
    input_name = landmark_session.get_inputs()[0].name
    output_names = [output.name for output in landmark_session.get_outputs()]
    detector = cv2.FaceDetectorYN.create(
        str(FACE_DETECTOR_MODEL),
        "",
        (320, 320),
        FACE_DETECTION_THRESHOLD,
        0.3,
        5000,
    )
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("No se pudo abrir la cámara 0.")

    print(
        f"Landmarks: {landmark_session.get_providers()[0]}, "
        f"adaptador DirectML {args.device_id}. "
        "Detector YuNet y captura de cámara usan CPU."
    )
    print("Pulsa 'q' para salir.")

    tracker = BlinkTracker()
    alarm_announced = False
    detector_input_size = None

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                raise RuntimeError("No se pudo leer un frame de la cámara.")

            height, width = frame.shape[:2]
            if detector_input_size != (width, height):
                detector.setInputSize((width, height))
                detector_input_size = (width, height)
            _, faces = detector.detect(frame)
            now = time.monotonic()

            if faces is None or len(faces) == 0:
                tracker.update(0.0, 0.0, now, False)
                points = None
            else:
                face = max(faces, key=lambda detection: detection[-1])
                points = infer_landmarks(
                    landmark_session,
                    input_name,
                    frame,
                    face[:4],
                    output_names,
                )
                if points is None:
                    tracker.update(0.0, 0.0, now, False)
                else:
                    left_ear = calculate_eye_aspect_ratio(points, LEFT_EYE_POINTS)
                    right_ear = calculate_eye_aspect_ratio(points, RIGHT_EYE_POINTS)
                    tracker.update(left_ear, right_ear, now, True)

            warning = tracker.active_elapsed >= TIMEOUT_SECONDS
            if warning and not alarm_announced:
                threading.Thread(target=speak_alarm, daemon=True).start()
                alarm_announced = True
            elif not warning and alarm_announced:
                alarm_announced = False

            if warning:
                status_text = f"ALERTA: Por favor parpadea | {tracker.active_elapsed:.1f}s"
                color = (0, 0, 255)
            elif points is None:
                status_text = "Rostro no detectado; temporizador pausado"
                color = (0, 165, 255)
            else:
                status_text = f"Parpadeos: {tracker.blink_count} | Tiempo: {tracker.active_elapsed:.1f}s"
                color = (255, 255, 255)

            if points is not None:
                eye_status = "Cerrados; tiempo pausado" if tracker.eyes_closed else "Abiertos"
                cv2.putText(
                    frame,
                    f"L EAR: {left_ear:.2f}  R EAR: {right_ear:.2f}  {eye_status}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

            if points is not None:
                for x, y in points:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
            cv2.imshow("BlinkEye AMD GPU", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
