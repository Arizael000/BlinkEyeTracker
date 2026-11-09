"""Alarma de voz con SAPI de Windows, sin bloquear el bucle de vídeo."""

from __future__ import annotations

import subprocess
import sys
import threading


def speak_alarm(message: str = "Parpadea, parpadea, parpadea") -> None:
    """Reproduce ``message`` con la voz en español de SAPI en un hilo aparte."""
    # En cadenas de PowerShell entre comillas simples, ' se escapa como ''.
    safe_message = message.replace("'", "''")

    powershell_command = (
        "$ErrorActionPreference = 'Stop'; "
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "try { "
        "$voice = $speaker.GetInstalledVoices() | "
        "Where-Object { $_.VoiceInfo.Culture.Name -like 'es-*' } | "
        "Select-Object -First 1; "
        "if ($voice) { $speaker.SelectVoice($voice.VoiceInfo.Name) }; "
        f"$speaker.Speak('{safe_message}') "
        "} "
        "finally { $speaker.Dispose() }"
    )

    def _run() -> None:
        try:
            subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    powershell_command,
                ],
                check=True,
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.CalledProcessError) as error:
            details = getattr(error, "stderr", None)
            print(f"No se pudo reproducir la alarma hablada: {details or error}", file=sys.stderr)

    threading.Thread(target=_run, daemon=True).start()
