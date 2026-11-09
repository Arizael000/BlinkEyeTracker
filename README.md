# BlinkEye Tracker

Aplicación de escritorio para Windows que detecta **parpadeos** y controla el **tiempo de uso de pantalla**, con una interfaz oscura moderna estilo Nvidia.

Detecta hasta **5 rostros** a la vez y avisa **con voz** si pasas demasiado tiempo sin parpadear.

## Requisitos

- Windows 10/11
- Python 3.10+ (probado con un entorno conda que ya trae MediaPipe)
- Cámara web
- El modelo `face_landmarker.task` en la misma carpeta del proyecto (ya incluido)

Instala las dependencias en tu entorno conda:

```powershell
conda activate <tu-entorno>
pip install -r requirements.txt
```

> Si usas un entorno que ya tiene `mediapipe`, `opencv` y `customtkinter`, puedes saltarte este paso.

## Ejecutar

```powershell
python BlinkEyeApp.py
```

## Interfaz

- **Barra superior**: título + botones de minimizar, maximizar y cerrar (ventana sin bordes; arrastra desde la barra para moverla, redimensiona desde los bordes y esquinas).
- **Panel izquierdo**: navegación entre *Monitor*, *Configuración* y *Acerca de*.
- **Panel derecho**:
  - Arriba: botones de control del vídeo (*Iniciar / Pausar / Detener*), interruptor de alarma y FPS.
  - Debajo: tarjetas con *rostros, parpadeos, tiempo sin parpadear y estado*, más un resumen por rostro.
  - Abajo: el **cuadro de vídeo** con los landmarks y el estado dibujados sobre cada rostro.

## Configuración (se guarda en `config.json`)

- Índice de cámara.
- Tiempo de alerta sin parpadear (1–15 s).
- Sensibilidad del parpadeo (0–100 %): a mayor valor, detecta cierres más sutiles.
- Brillo virtual (0–100 %): aumenta el brillo de la imagen en ambientes de poca luz.
- Activar/desactivar la alarma de voz.
- Mostrar/ocultar los landmarks sobre el vídeo.

## Empaquetar como .exe (opcional)

```powershell
pip install pyinstaller
pyinstaller --noconsole --onefile --add-data "face_landmarker.task;." BlinkEyeApp.py
```

## Estructura

```
blinkEye/
  BlinkEyeApp.py        # Interfaz (CustomTkinter, barra de título, paneles)
  core/
    engine.py           # Hilo de cámara + MediaPipe + seguimiento
    tracker.py          # Lógica de parpadeos (EAR) reutilizada
    alarm.py            # Alarma de voz (SAPI de Windows)
    settings.py         # Configuración persistente (config.json)
    theme.py            # Paleta de colores oscura
  face_landmarker.task  # Modelo MediaPipe
  requirements.txt
```

## Créditos

Basado en la lógica de `parpadeo_multiple.py` (detección multi-rostro con MediaPipe Face Landmarker).
