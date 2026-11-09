"""Paleta de colores y constantes de estilo de BlinkEye (tema oscuro estilo Nvidia)."""

from __future__ import annotations

# --- Colores base ---
BG = "#1B1B1B"            # Fondo del contenido principal
SIDEBAR = "#151515"       # Panel lateral izquierdo
TITLEBAR = "#141414"      # Barra de título personalizada
CARD = "#232323"          # Tarjetas y encabezados
CARD_HOVER = "#2D2D2D"    # Hover de tarjetas / botones neutros
BORDER = "#2E2E2E"        # Bordes sutiles
VIDEO_BG = "#000000"      # Fondo del cuadro de video

# --- Acento (verde Nvidia) ---
ACCENT = "#76B900"
ACCENT_HOVER = "#8FD01F"
ACCENT_TEXT = "#101010"

# --- Texto ---
TEXT = "#E8E8E8"
TEXT_DIM = "#9A9A9A"
TEXT_MUTED = "#6E6E6E"

# --- Estados ---
OK = "#76B900"
WARNING = "#FB8C00"
DANGER = "#E53935"
INFO = "#29B6F6"

# --- Tipografías ---
FONT_FAMILY = "Segoe UI"


def tc(hex_color: str) -> tuple[str, str]:
    """Convierte un color a la tupla (claro, oscuro) que esperan los widgets de CTk."""
    return (hex_color, hex_color)
