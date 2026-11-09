"""BlinkEye Tracker — Interfaz gráfica moderna (estilo Nvidia) para Windows.

Ventana sin bordes con barra de título personalizada, panel lateral de opciones,
controles de vídeo, información de parpadeos en tarjetas y cuadro de vídeo.
El procesamiento (cámara + MediaPipe) corre en un hilo aparte.
"""

from __future__ import annotations

import ctypes
import queue
import tkinter as tk
from ctypes import wintypes
from pathlib import Path

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from core import theme as T
from core.settings import Settings

MIN_WIDTH = 940
MIN_HEIGHT = 620


def fit_frame(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Ajusta un frame al tamaño objetivo conservando la relación de aspecto (letterbox)."""
    if target_w < 10 or target_h < 10:
        return frame
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return frame
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


class StatTile(ctk.CTkFrame):
    """Tarjeta pequeña con una etiqueta y un valor grande."""

    def __init__(self, master, caption: str, value: str = "—"):
        super().__init__(master, fg_color=T.CARD, corner_radius=10)
        ctk.CTkLabel(
            self, text=caption.upper(), text_color=T.TEXT_MUTED,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=11, weight="bold"),
        ).pack(anchor="w", padx=14, pady=(10, 0))
        self.value = ctk.CTkLabel(
            self, text=value, text_color=T.TEXT,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=22, weight="bold"),
        )
        self.value.pack(anchor="w", padx=14, pady=(0, 10))

    def set_value(self, text: str, color: str | None = None) -> None:
        self.value.configure(text=text, text_color=color or T.TEXT)


class NavButton(ctk.CTkButton):
    """Botón de navegación del panel lateral, resaltado cuando está activo."""

    def __init__(self, master, text: str, command):
        super().__init__(
            master, text=text, command=command, anchor="w", height=42,
            corner_radius=8, fg_color="transparent", hover_color=T.CARD_HOVER,
            text_color=T.TEXT_DIM, font=ctk.CTkFont(family=T.FONT_FAMILY, size=14),
        )

    def set_active(self, active: bool) -> None:
        if active:
            self.configure(
                fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER, text_color=T.ACCENT_TEXT,
            )
        else:
            self.configure(
                fg_color="transparent", hover_color=T.CARD_HOVER, text_color=T.TEXT_DIM,
            )


class TitleBar(ctk.CTkFrame):
    """Barra de título personalizada con arrastre y botones de ventana."""

    def __init__(self, master, on_min, on_max, on_close):
        super().__init__(master, height=42, fg_color=T.TITLEBAR, corner_radius=0)
        self.pack_propagate(False)
        self.grid_columnconfigure(0, weight=1)
        self._drag_offset = (0, 0)

        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=0, column=0, sticky="w", padx=(16, 0))
        ctk.CTkLabel(
            left, text="◉", text_color=T.ACCENT,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=20),
        ).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(
            left, text="BlinkEye Tracker", text_color=T.TEXT,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=13, weight="bold"),
        ).pack(side="left")

        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=0, column=1, sticky="e")

        ctk.CTkButton(
            right, text="—", width=46, height=42, corner_radius=0,
            fg_color="transparent", hover_color="#2A2A2A", text_color=T.TEXT,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=14), command=on_min,
        ).pack(side="left", fill="y")
        self.btn_max = ctk.CTkButton(
            right, text="▢", width=46, height=42, corner_radius=0,
            fg_color="transparent", hover_color="#2A2A2A", text_color=T.TEXT,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=13), command=on_max,
        )
        self.btn_max.pack(side="left", fill="y")
        ctk.CTkButton(
            right, text="✕", width=46, height=42, corner_radius=0,
            fg_color="transparent", hover_color="#E81123", text_color=T.TEXT,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=14), command=on_close,
        ).pack(side="left", fill="y")

        # Arrastre de la ventana desde la barra de título.
        for widget in [self, left, *left.winfo_children()]:
            widget.bind("<Button-1>", self._start_move)
            widget.bind("<B1-Motion>", self._on_move)
            widget.bind("<Double-1>", lambda _e: on_max())

    def set_max_icon(self, maximized: bool) -> None:
        self.btn_max.configure(text="❐" if maximized else "▢")

    def _start_move(self, event):
        toplevel = self.winfo_toplevel()
        self._drag_offset = (
            event.x_root - toplevel.winfo_x(),
            event.y_root - toplevel.winfo_y(),
        )

    def _on_move(self, event):
        toplevel = self.winfo_toplevel()
        x = event.x_root - self._drag_offset[0]
        y = event.y_root - self._drag_offset[1]
        toplevel.geometry(f"+{x}+{y}")


class BlinkEyeApp(ctk.CTk):
    """Ventana principal de BlinkEye."""

    def __init__(self, worker_cls, model_path: Path):
        super().__init__()
        self.worker_cls = worker_cls
        self.model_path = model_path
        self.settings = Settings.load()

        self.worker = None
        self._current_photo = None
        self._maximized = False
        self._saved_geometry = None
        self._resize_dir = None
        self._resize_start = (0, 0)
        self._resize_geom = (0, 0, 0, 0)
        self._current_page = "monitor"

        ctk.set_appearance_mode("dark")
        self.configure(fg_color=T.BG)
        self.title("BlinkEye Tracker")
        self.geometry("1180x760")
        self.minsize(MIN_WIDTH, MIN_HEIGHT)
        self._center(1180, 760)

        self._build_ui()
        self._sync_settings_widgets()
        self._show_page("monitor")
        self._set_status("Listo. Pulsa Iniciar para comenzar.")

        # Ventana sin bordes + presencia en la barra de tareas de Windows.
        self.overrideredirect(True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(60, self._apply_taskbar_style)
        self.after(15, self._poll_frames)

    # ------------------------------------------------------------------ #
    #  Construcción de la interfaz
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self._build_titlebar()
        self._build_body()

    def _build_titlebar(self) -> None:
        self.titlebar = TitleBar(
            self, self._minimize, self._toggle_maximize, self._on_close,
        )
        self.titlebar.pack(side="top", fill="x")

    def _build_body(self) -> None:
        self.body = ctk.CTkFrame(self, fg_color=T.BG, corner_radius=0)
        self.body.pack(side="top", fill="both", expand=True)

        self._build_sidebar(self.body)
        self.content = ctk.CTkFrame(self.body, fg_color=T.BG, corner_radius=0)
        self.content.pack(side="left", fill="both", expand=True)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        # Asegura que el contenido se reajuste al redimensionar la ventana.
        self.content.bind("<Configure>", self._on_content_resize)

        self.pages = {}
        self.pages["monitor"] = self._build_monitor_page(self.content)
        self.pages["settings"] = self._build_settings_page(self.content)
        self.pages["about"] = self._build_about_page(self.content)
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")

        self._build_statusbar()
        self._add_resize_grips()

    def _on_content_resize(self, event) -> None:
        """Fuerza actualización de layout cuando el contenedor de páginas cambia de tamaño."""
        if event.widget is self.content:
            self.content.update_idletasks()

    def _build_sidebar(self, body) -> None:
        sidebar = ctk.CTkFrame(body, width=212, fg_color=T.SIDEBAR, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ctk.CTkLabel(
            sidebar, text="BlinkEye", text_color=T.TEXT,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=20, weight="bold"),
        ).pack(anchor="w", padx=20, pady=(24, 0))
        ctk.CTkLabel(
            sidebar, text="Monitor de parpadeos", text_color=T.TEXT_MUTED,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=12),
        ).pack(anchor="w", padx=20, pady=(0, 22))

        self.nav_buttons = {}
        self.nav_buttons["monitor"] = NavButton(
            sidebar, "▣  Monitor", lambda: self._show_page("monitor"),
        )
        self.nav_buttons["settings"] = NavButton(
            sidebar, "⚙  Configuración", lambda: self._show_page("settings"),
        )
        self.nav_buttons["about"] = NavButton(
            sidebar, "ⓘ  Acerca de", lambda: self._show_page("about"),
        )
        for btn in self.nav_buttons.values():
            btn.pack(fill="x", padx=12, pady=3)

        ctk.CTkLabel(
            sidebar, text="v1.0.0", text_color=T.TEXT_MUTED,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=11),
        ).pack(side="bottom", anchor="w", padx=20, pady=16)

    # --- Página: Monitor -------------------------------------------------
    def _build_monitor_page(self, parent) -> ctk.CTkFrame:
        page = ctk.CTkFrame(parent, fg_color=T.BG, corner_radius=0)
        page.grid_rowconfigure(2, weight=1)
        page.grid_columnconfigure(0, weight=1)

        # Barra de controles.
        bar = ctk.CTkFrame(page, fg_color=T.CARD, corner_radius=12)
        bar.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 6))
        bar.grid_columnconfigure(0, weight=1)  # grupo izquierdo expansible
        bar.grid_columnconfigure(1, weight=0)  # grupo derecho fijo

        # --- Grupo IZQUIERDO: botones de control ---
        left_group = ctk.CTkFrame(bar, fg_color="transparent")
        left_group.grid(row=0, column=0, sticky="w", padx=(14, 0), pady=6)

        self.btn_start = ctk.CTkButton(
            left_group, text="▶  Iniciar", command=self._start_camera, height=34,
            corner_radius=8, fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER,
            text_color=T.ACCENT_TEXT, font=ctk.CTkFont(family=T.FONT_FAMILY, size=13, weight="bold"),
        )
        self.btn_start.pack(side="left", padx=(0, 6))
        self.btn_pause = ctk.CTkButton(
            left_group, text="⏸  Pausar", command=self._toggle_pause, height=34,
            corner_radius=8, fg_color=T.CARD_HOVER, hover_color="#383838",
            text_color=T.TEXT, font=ctk.CTkFont(family=T.FONT_FAMILY, size=13),
            state="disabled",
        )
        self.btn_pause.pack(side="left", padx=6)
        ctk.CTkButton(
            left_group, text="⏹  Detener", command=self._stop_camera, height=34,
            corner_radius=8, fg_color=T.CARD_HOVER, hover_color="#383838",
            text_color=T.TEXT, font=ctk.CTkFont(family=T.FONT_FAMILY, size=13),
        ).pack(side="left", padx=6)

        ctk.CTkFrame(left_group, width=1, fg_color=T.BORDER).pack(side="left", fill="y", padx=12, pady=4)

        self.switch_alarm_ctrl = ctk.CTkSwitch(
            left_group, text="Alarma", command=lambda: self._on_alarm_changed(self.switch_alarm_ctrl),
            progress_color=T.ACCENT, text_color=T.TEXT_DIM,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=13),
        )
        self.switch_alarm_ctrl.pack(side="left", padx=6)

        # --- Grupo DERECHO: FPS + Sliders apilados verticalmente ---
        right_group = ctk.CTkFrame(bar, fg_color="transparent")
        right_group.grid(row=0, column=1, sticky="e", padx=(0, 14), pady=6)

        self.fps_label = ctk.CTkLabel(
            right_group, text="-- FPS", text_color=T.TEXT_DIM,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=12),
        )
        self.fps_label.pack(anchor="e", pady=(0, 4))

        # Contenedor vertical para los dos sliders.
        sliders_box = ctk.CTkFrame(right_group, fg_color="transparent")
        sliders_box.pack(anchor="e")

        # Sensibilidad.
        sens_row = ctk.CTkFrame(sliders_box, fg_color="transparent")
        sens_row.pack(anchor="e", pady=2)
        ctk.CTkLabel(
            sens_row, text="Sensibilidad", text_color=T.TEXT_DIM,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=11),
        ).pack(side="left", padx=(0, 6))
        self.sensitivity_slider_monitor = ctk.CTkSlider(
            sens_row, from_=0, to=100, number_of_steps=100, width=120, height=18,
            command=self._on_sensitivity_changed, progress_color=T.ACCENT,
            button_color=T.ACCENT, button_hover_color=T.ACCENT_HOVER,
        )
        self.sensitivity_slider_monitor.pack(side="left", padx=(0, 6))
        self.sensitivity_value_monitor = ctk.CTkLabel(
            sens_row, text="50%", text_color=T.ACCENT,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=11, weight="bold"), width=36,
        )
        self.sensitivity_value_monitor.pack(side="left")

        # Brillo.
        bright_row = ctk.CTkFrame(sliders_box, fg_color="transparent")
        bright_row.pack(anchor="e", pady=2)
        ctk.CTkLabel(
            bright_row, text="Brillo", text_color=T.TEXT_DIM,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=11),
        ).pack(side="left", padx=(0, 6))
        self.brightness_slider_monitor = ctk.CTkSlider(
            bright_row, from_=0, to=100, number_of_steps=100, width=120, height=18,
            command=self._on_brightness_changed, progress_color=T.ACCENT,
            button_color=T.ACCENT, button_hover_color=T.ACCENT_HOVER,
        )
        self.brightness_slider_monitor.pack(side="left", padx=(0, 6))
        self.brightness_value_monitor = ctk.CTkLabel(
            bright_row, text="0%", text_color=T.ACCENT,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=11, weight="bold"), width=36,
        )
        self.brightness_value_monitor.pack(side="left")

        # Tarjetas de información.
        info = ctk.CTkFrame(page, fg_color="transparent")
        info.grid(row=1, column=0, sticky="ew", padx=14, pady=6)
        for i in range(4):
            info.grid_columnconfigure(i, weight=1, uniform="tiles")

        self.tile_faces = StatTile(info, "Rostros")
        self.tile_blinks = StatTile(info, "Parpadeos")
        self.tile_time = StatTile(info, "Sin parpadear", "0.0s")
        self.tile_state = StatTile(info, "Estado")
        tiles = [self.tile_faces, self.tile_blinks, self.tile_time, self.tile_state]
        for i, tile in enumerate(tiles):
            tile.grid(row=0, column=i, sticky="ew", padx=(0, 8) if i < 3 else 0)

        self.face_summary = ctk.CTkLabel(
            info, text="", anchor="w", text_color=T.TEXT_DIM,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=12), wraplength=0,
        )
        self.face_summary.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        self.face_summary.bind("<Configure>", lambda e: self.face_summary.configure(wraplength=e.width))

        # Cuadro de vídeo (tk.Label nativo para máxima fluidez al mostrar frames).
        self.video_label = tk.Label(
            page, bg=T.VIDEO_BG, fg=T.TEXT_DIM, bd=0, highlightthickness=0,
            text="Pulsa \"Iniciar\" para comenzar", font=(T.FONT_FAMILY, 15),
        )
        self.video_label.grid(row=2, column=0, sticky="nsew", padx=14, pady=(6, 14))
        return page

    # --- Página: Configuración ------------------------------------------
    def _build_settings_page(self, parent) -> ctk.CTkFrame:
        page = ctk.CTkScrollableFrame(parent, fg_color=T.BG, corner_radius=0)
        page.grid_columnconfigure(0, weight=1)

        self._section_title(page, "Configuración", "Los cambios se aplican al instante.")

        self._section_header(page, "Cámara")
        self.cam_menu = ctk.CTkOptionMenu(
            page, values=["0", "1", "2", "3"], command=self._on_camera_changed,
            fg_color=T.CARD, button_color=T.CARD_HOVER, button_hover_color="#383838",
            text_color=T.TEXT, font=ctk.CTkFont(family=T.FONT_FAMILY, size=13),
        )
        self.cam_menu.pack(anchor="w", padx=20, pady=(0, 14))

        self._section_header(page, "Detección")
        row = ctk.CTkFrame(page, fg_color="transparent")
        row.pack(anchor="w", fill="x", padx=20, pady=(0, 6))
        ctk.CTkLabel(
            row, text="Tiempo de alerta", text_color=T.TEXT_DIM,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=13),
        ).pack(side="left")
        self.timeout_value = ctk.CTkLabel(
            row, text="5.0 s", text_color=T.ACCENT,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=13, weight="bold"),
        )
        self.timeout_value.pack(side="right")

        self.timeout_slider = ctk.CTkSlider(
            page, from_=1, to=15, number_of_steps=28, command=self._on_timeout_changed,
            progress_color=T.ACCENT, button_color=T.ACCENT, button_hover_color=T.ACCENT_HOVER,
        )
        self.timeout_slider.pack(fill="x", padx=20, pady=(0, 14))

        sens_row = ctk.CTkFrame(page, fg_color="transparent")
        sens_row.pack(anchor="w", fill="x", padx=20, pady=(0, 6))
        ctk.CTkLabel(
            sens_row, text="Sensibilidad (mayor = detecta blinks más sutiles)",
            text_color=T.TEXT_DIM, font=ctk.CTkFont(family=T.FONT_FAMILY, size=13),
        ).pack(side="left")
        self.sensitivity_value = ctk.CTkLabel(
            sens_row, text="50%", text_color=T.ACCENT,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=13, weight="bold"),
        )
        self.sensitivity_value.pack(side="right")

        self.sensitivity_slider = ctk.CTkSlider(
            page, from_=0, to=100, number_of_steps=100, command=self._on_sensitivity_changed,
            progress_color=T.ACCENT, button_color=T.ACCENT, button_hover_color=T.ACCENT_HOVER,
        )
        self.sensitivity_slider.pack(fill="x", padx=20, pady=(0, 14))

        self._section_header(page, "Imagen")
        bright_row = ctk.CTkFrame(page, fg_color="transparent")
        bright_row.pack(anchor="w", fill="x", padx=20, pady=(0, 6))
        ctk.CTkLabel(
            bright_row, text="Brillo virtual (poca luz)",
            text_color=T.TEXT_DIM, font=ctk.CTkFont(family=T.FONT_FAMILY, size=13),
        ).pack(side="left")
        self.brightness_value = ctk.CTkLabel(
            bright_row, text="0%", text_color=T.ACCENT,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=13, weight="bold"),
        )
        self.brightness_value.pack(side="right")

        self.brightness_slider = ctk.CTkSlider(
            page, from_=0, to=100, number_of_steps=100, command=self._on_brightness_changed,
            progress_color=T.ACCENT, button_color=T.ACCENT, button_hover_color=T.ACCENT_HOVER,
        )
        self.brightness_slider.pack(fill="x", padx=20, pady=(0, 14))

        self._section_header(page, "Alertas y visualización")
        self.switch_alarm_settings = ctk.CTkSwitch(
            page, text="Alarma de voz (cuando no parpadeas)",
            command=lambda: self._on_alarm_changed(self.switch_alarm_settings),
            progress_color=T.ACCENT, text_color=T.TEXT_DIM,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=13),
        )
        self.switch_alarm_settings.pack(anchor="w", padx=20, pady=(0, 10))

        self.switch_landmarks = ctk.CTkSwitch(
            page, text="Mostrar landmarks en el vídeo",
            command=self._on_landmarks_changed,
            progress_color=T.ACCENT, text_color=T.TEXT_DIM,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=13),
        )
        self.switch_landmarks.pack(anchor="w", padx=20, pady=(0, 16))

        ctk.CTkLabel(
            page,
            text="Consejo: la alarma avisa con voz cuando algún rostro supera el\n"
                 "tiempo de alerta sin parpadear.",
            text_color=T.TEXT_MUTED, justify="left", anchor="w",
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=12),
        ).pack(anchor="w", padx=20, pady=(0, 20))
        return page

    # --- Página: Acerca de ----------------------------------------------
    def _build_about_page(self, parent) -> ctk.CTkFrame:
        page = ctk.CTkFrame(parent, fg_color=T.BG, corner_radius=0)
        page.grid_columnconfigure(0, weight=1)

        card = ctk.CTkFrame(page, fg_color=T.CARD, corner_radius=12)
        card.grid(row=0, column=0, padx=40, pady=40, sticky="ew")

        ctk.CTkLabel(
            card, text="◉", text_color=T.ACCENT,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=48),
        ).pack(pady=(24, 8))
        ctk.CTkLabel(
            card, text="BlinkEye Tracker", text_color=T.TEXT,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=22, weight="bold"),
        ).pack()
        ctk.CTkLabel(
            card, text="v1.0.0", text_color=T.TEXT_MUTED,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=13),
        ).pack(pady=(0, 16))

        lines = [
            "Detector de parpadeos y seguimiento del tiempo de uso de pantalla.",
            "Detecta hasta 5 rostros y avisa con voz si pasas mucho tiempo",
            "sin parpadear, para cuidar la salud de tus ojos.",
            "",
            "Motor de visión: MediaPipe Face Landmarker",
            "Interfaz: CustomTkinter (tema oscuro estilo Nvidia)",
        ]
        for line in lines:
            ctk.CTkLabel(
                card, text=line, text_color=T.TEXT_DIM,
                font=ctk.CTkFont(family=T.FONT_FAMILY, size=13),
            ).pack(pady=1)

        ctk.CTkLabel(
            card, text="", text_color=T.TEXT_MUTED,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=11),
        ).pack(pady=(12, 24))
        return page

    def _build_statusbar(self) -> None:
        self.statusbar = ctk.CTkFrame(self, height=28, fg_color=T.TITLEBAR, corner_radius=0)
        self.statusbar.pack(side="bottom", fill="x")
        self.statusbar.pack_propagate(False)
        self.statusbar.grid_columnconfigure(0, weight=1)
        self.statusbar.grid_columnconfigure(1, weight=0)
        self.status_label = ctk.CTkLabel(
            self.statusbar, text="", anchor="w", text_color=T.TEXT_MUTED,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=11), wraplength=0,
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=12)
        self.status_label.bind("<Configure>", lambda e: self.status_label.configure(wraplength=e.width))
        self.backend_label = ctk.CTkLabel(
            self.statusbar, text="MediaPipe", text_color=T.TEXT_MUTED,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=11),
        )
        self.backend_label.grid(row=0, column=1, sticky="e", padx=12)

    def _add_resize_grips(self) -> None:
        """Bordes de redimensionado manuales (la ventana no tiene bordes nativos)."""
        # (cursor, dirección, padre, ancho, alto, opciones de place)
        specs = [
            # Bordes izquierdo/derecho sobre el cuerpo, sin tapar la barra de título.
            ("sb_h_double_arrow", "e", self.body, 6, None, {"relx": 1.0, "rely": 0.0, "relheight": 1.0, "anchor": "ne"}),
            ("sb_h_double_arrow", "w", self.body, 6, None, {"relx": 0.0, "rely": 0.0, "relheight": 1.0, "anchor": "nw"}),
            # Borde inferior y esquinas sobre la ventana completa.
            ("sb_v_double_arrow", "s", self, None, 6, {"relx": 0.0, "rely": 1.0, "relwidth": 1.0, "anchor": "sw"}),
            ("bottom_right_corner", "se", self, 16, 16, {"relx": 1.0, "rely": 1.0, "anchor": "se"}),
            ("bottom_left_corner", "sw", self, 16, 16, {"relx": 0.0, "rely": 1.0, "anchor": "sw"}),
        ]
        for cursor, direction, parent, width, height, place_kwargs in specs:
            ctor_kwargs = {"fg_color": T.BG, "corner_radius": 0, "cursor": cursor}
            if width is not None:
                ctor_kwargs["width"] = width
            if height is not None:
                ctor_kwargs["height"] = height
            grip = ctk.CTkFrame(parent, **ctor_kwargs)
            grip.place(**place_kwargs)
            grip.bind("<Button-1>", lambda e, d=direction: self._start_resize(e, d))
            grip.bind("<B1-Motion>", self._on_resize)

    # ------------------------------------------------------------------ #
    #  Helpers de la interfaz
    # ------------------------------------------------------------------ #
    def _section_title(self, page, text, subtitle=None) -> None:
        ctk.CTkLabel(
            page, text=text, text_color=T.TEXT,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=20, weight="bold"),
        ).pack(anchor="w", padx=20, pady=(24, 2))
        if subtitle:
            ctk.CTkLabel(
                page, text=subtitle, text_color=T.TEXT_MUTED,
                font=ctk.CTkFont(family=T.FONT_FAMILY, size=12),
            ).pack(anchor="w", padx=20, pady=(0, 16))

    def _section_header(self, page, text) -> None:
        ctk.CTkLabel(
            page, text=text, text_color=T.TEXT_DIM,
            font=ctk.CTkFont(family=T.FONT_FAMILY, size=14, weight="bold"),
        ).pack(anchor="w", padx=20, pady=(8, 8))

    def _show_page(self, name: str) -> None:
        self._current_page = name
        self.pages[name].lift()
        for key, btn in self.nav_buttons.items():
            btn.set_active(key == name)

    def _center(self, w: int, h: int) -> None:
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(20, (sh - h) // 3)}")

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.configure(
            text=text, text_color=T.DANGER if error else T.TEXT_MUTED,
        )

    def _sync_settings_widgets(self) -> None:
        self.timeout_slider.set(self.settings.timeout_seconds)
        self.timeout_value.configure(text=f"{self.settings.timeout_seconds:.1f} s")
        self.sensitivity_slider.set(self.settings.sensitivity)
        self.sensitivity_value.configure(text=f"{self.settings.sensitivity}%")
        self.sensitivity_slider_monitor.set(self.settings.sensitivity)
        self.sensitivity_value_monitor.configure(text=f"{self.settings.sensitivity}%")
        self.brightness_slider.set(self.settings.brightness)
        self.brightness_value.configure(text=f"{self.settings.brightness}%")
        self.brightness_slider_monitor.set(self.settings.brightness)
        self.brightness_value_monitor.configure(text=f"{self.settings.brightness}%")
        self.cam_menu.set(str(self.settings.camera_index))
        for sw in (self.switch_alarm_ctrl, self.switch_alarm_settings):
            sw.select() if self.settings.alarm_enabled else sw.deselect()
        self.switch_landmarks.select() if self.settings.show_landmarks else self.switch_landmarks.deselect()

    # ------------------------------------------------------------------ #
    #  Ventana (barra de título personalizada)
    # ------------------------------------------------------------------ #
    def _apply_taskbar_style(self) -> None:
        """Reañade WS_EX_APPWINDOW para que la ventana aparezca en la barra de tareas."""
        try:
            self.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            GWL_EXSTYLE = -20
            WS_EX_APPWINDOW = 0x00040000
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_APPWINDOW)
        except Exception:
            pass

    def _work_area(self) -> tuple[int, int, int, int]:
        try:
            rect = wintypes.RECT()
            ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
            return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
        except Exception:
            return 0, 0, self.winfo_screenwidth(), self.winfo_screenheight()

    def _minimize(self) -> None:
        self.iconify()

    def _toggle_maximize(self) -> None:
        if self._maximized:
            if self._saved_geometry:
                self.geometry(self._saved_geometry)
            self._maximized = False
        else:
            self._saved_geometry = self.geometry()
            x, y, w, h = self._work_area()
            self.geometry(f"{w}x{h}+{x}+{y}")
            self._maximized = True
        self.titlebar.set_max_icon(self._maximized)

    def _start_resize(self, event, direction: str) -> None:
        self._resize_dir = direction
        self._resize_start = (event.x_root, event.y_root)
        self._resize_geom = (
            self.winfo_x(), self.winfo_y(), self.winfo_width(), self.winfo_height(),
        )

    def _on_resize(self, event) -> None:
        if not self._resize_dir:
            return
        dx = event.x_root - self._resize_start[0]
        dy = event.y_root - self._resize_start[1]
        x, y, w, h = self._resize_geom
        d = self._resize_dir
        if "e" in d:
            w += dx
        if "s" in d:
            h += dy
        if "w" in d:
            w -= dx
            x += dx
        if "n" in d:
            h -= dy
            y += dy
        if w < MIN_WIDTH:
            if "w" in d:
                x -= (MIN_WIDTH - w)
            w = MIN_WIDTH
        if h < MIN_HEIGHT:
            if "n" in d:
                y -= (MIN_HEIGHT - h)
            h = MIN_HEIGHT
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _on_close(self) -> None:
        self._stop_camera()
        self.settings.save()
        self.destroy()

    # ------------------------------------------------------------------ #
    #  Control de la cámara
    # ------------------------------------------------------------------ #
    def _start_camera(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        if not self.model_path.is_file():
            self._set_status(
                f"⚠ No se encontró '{self.model_path.name}' en la carpeta del proyecto.", error=True,
            )
            return
        self._clear_video()
        self.worker = self.worker_cls(self.settings)
        self.worker.start()
        self.btn_start.configure(state="disabled")
        self.btn_pause.configure(state="normal", text="⏸  Pausar")
        self._set_status(f"Iniciando cámara {self.settings.camera_index}…")

    def _stop_camera(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self._clear_video()
        self.btn_start.configure(state="normal")
        self.btn_pause.configure(state="disabled", text="⏸  Pausar")
        self._set_status("Cámara detenida")

    def _toggle_pause(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            return
        paused = not self.worker.paused
        self.worker.set_paused(paused)
        self.btn_pause.configure(text="▶  Reanudar" if paused else "⏸  Pausar")
        self._set_status("Vídeo en pausa" if paused else "Vídeo en reproducción")

    def _restart_camera(self) -> None:
        was_running = self.worker is not None and self.worker.is_alive()
        if was_running:
            self._stop_camera()
        if was_running:
            self._start_camera()

    # ------------------------------------------------------------------ #
    #  Callbacks de configuración
    # ------------------------------------------------------------------ #
    def _on_camera_changed(self, value: str) -> None:
        self.settings.camera_index = int(value)
        self.settings.save()
        self._restart_camera()
        self._set_status(f"Cámara cambiada a índice {value}")

    def _on_timeout_changed(self, value: float) -> None:
        self.settings.timeout_seconds = round(float(value), 1)
        self.timeout_value.configure(text=f"{self.settings.timeout_seconds:.1f} s")
        self.settings.save()

    def _on_sensitivity_changed(self, value: float) -> None:
        self.settings.sensitivity = int(round(float(value)))
        self.sensitivity_value.configure(text=f"{self.settings.sensitivity}%")
        self.sensitivity_value_monitor.configure(text=f"{self.settings.sensitivity}%")
        self.settings.save()

    def _on_brightness_changed(self, value: float) -> None:
        self.settings.brightness = int(round(float(value)))
        self.brightness_value.configure(text=f"{self.settings.brightness}%")
        self.brightness_value_monitor.configure(text=f"{self.settings.brightness}%")
        self.settings.save()

    def _on_alarm_changed(self, source) -> None:
        self.settings.alarm_enabled = bool(source.get())
        for sw in (self.switch_alarm_ctrl, self.switch_alarm_settings):
            sw.select() if self.settings.alarm_enabled else sw.deselect()
        self.settings.save()

    def _on_landmarks_changed(self) -> None:
        self.settings.show_landmarks = bool(self.switch_landmarks.get())
        self.settings.save()

    # ------------------------------------------------------------------ #
    #  Bucle de renderizado (hilo principal)
    # ------------------------------------------------------------------ #
    def _poll_frames(self) -> None:
        worker = self.worker
        last = None
        if worker is not None:
            try:
                while True:
                    last = worker.frame_queue.get_nowait()
            except queue.Empty:
                pass

            if not worker.is_alive():
                error = worker.error
                self.worker = None
                self.btn_start.configure(state="normal")
                self.btn_pause.configure(state="disabled", text="⏸  Pausar")
                if error:
                    self._set_status(f"⚠ {error}", error=True)
                else:
                    self._set_status("Cámara detenida")
                self._clear_video()

        if last is not None and self._current_page == "monitor":
            self._render_frame(*last)

        self.after(15, self._poll_frames)

    def _render_frame(self, frame: np.ndarray, stats: dict) -> None:
        lbl_w = self.video_label.winfo_width()
        lbl_h = self.video_label.winfo_height()
        if lbl_w < 10 or lbl_h < 10:
            lbl_w, lbl_h = 960, 560

        fitted = fit_frame(frame, lbl_w, lbl_h)
        # Si está pausado, añadir indicador visual semi-transparente.
        if stats.get("paused"):
            overlay = fitted.copy()
            cv2.rectangle(overlay, (0, 0), (lbl_w, lbl_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.35, fitted, 0.65, 0, fitted)
            cv2.putText(
                fitted, "⏸ PAUSADO", (lbl_w // 2 - 80, lbl_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
            )
        rgb = cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.configure(image=photo, text="")
        self._current_photo = photo

        self._update_info(stats)

    def _update_info(self, stats: dict) -> None:
        self.tile_faces.set_value(str(stats["num_faces"]))
        self.tile_blinks.set_value(str(stats["total_blinks"]))
        self.tile_time.set_value(f"{stats['max_elapsed']:.1f}s")

        if stats["num_faces"] == 0:
            state, color = "SIN ROSTRO", T.WARNING
        elif stats["global_warning"]:
            state, color = "ALERTA", T.DANGER
        else:
            state, color = "NORMAL", T.OK
        self.tile_state.set_value(state, color=color)

        if stats["faces"]:
            parts = []
            for f in stats["faces"]:
                eyes = "Cerrados" if f["eyes_closed"] else "Abiertos"
                parts.append(
                    f"R{f['index']}: {f['blinks']} parpadeos · {f['elapsed']:.1f}s · {eyes}"
                )
            self.face_summary.configure(text="   |   ".join(parts), text_color=T.TEXT_DIM)
        else:
            self.face_summary.configure(
                text="Sin rostro detectado · el temporizador está pausado", text_color=T.TEXT_MUTED,
            )
        self.fps_label.configure(text=f"{stats.get('fps', 0):.0f} FPS")

    def _clear_video(self) -> None:
        self.video_label.configure(
            image="", text="Pulsa \"Iniciar\" para comenzar", fg=T.TEXT_DIM,
        )
        self._current_photo = None
        self.tile_faces.set_value("0")
        self.tile_blinks.set_value("0")
        self.tile_time.set_value("0.0s")
        self.tile_state.set_value("—", color=T.TEXT_MUTED)
        self.face_summary.configure(text="", text_color=T.TEXT_MUTED)
        self.fps_label.configure(text="-- FPS")


def _show_dependency_error(package: str, exc: BaseException) -> None:
    try:
        import tkinter.messagebox as mb
        root = tk.Tk()
        root.withdraw()
        mb.showerror(
            "BlinkEye — dependencia faltante",
            f"No se pudo importar '{package}'.\n\n{exc}\n\n"
            "Instálalo en tu entorno de conda, por ejemplo:\n"
            "pip install mediapipe\n",
        )
        root.destroy()
    except Exception:
        print(f"Falta una dependencia ({package}): {exc}")


def main() -> int:
    try:
        from core.engine import CameraWorker, MODEL_PATH  # noqa: F401
    except ImportError as exc:
        _show_dependency_error("mediapipe", exc)
        return 1

    app = BlinkEyeApp(CameraWorker, MODEL_PATH)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
