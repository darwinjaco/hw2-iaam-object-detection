"""
Configuración centralizada del sistema de monitoreo por ROI.

Todos los parámetros del pipeline viven aquí para que los scripts
(run_pipeline.py, app.py, roi_editor.py) compartan una única fuente de verdad.
Las ROIs se persisten en config/rois.json en coordenadas NORMALIZADAS [0,1],
de modo que la configuración es independiente de la resolución del video.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------
# Rutas del proyecto
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_IN_DEFAULT = BASE_DIR / "videos" / "input" / "Track.mp4"
VIDEO_OUT_DEFAULT = BASE_DIR / "videos" / "output" / "roi_counting.mp4"
VIDEO_OUT_BASE = BASE_DIR / "videos" / "output" / "tracking_base.mp4"
OUTPUT_DIR = BASE_DIR / "outputs"
ROIS_FILE_DEFAULT = BASE_DIR / "config" / "rois.json"

# --------------------------------------------------------------------------
# Modelo y tracker
# --------------------------------------------------------------------------
MODEL_NAME = "yolo26m.pt"
TRACKER_CFG = "bytetrack.yaml"      # requisito explícito de la tarea

# Clases COCO de interés. El modelo preentrenado usa el vocabulario COCO-80.
COCO_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TARGET_CLASSES: list[int] = [2, 3, 5, 7]

# Umbrales de inferencia. Se declaran explícitamente en lugar de heredar los
# defaults de Ultralytics para que el informe pueda reportarlos.
CONF_THRES = 0.35
IMGSZ = 1280        # el video es 1080p; 1280 conserva más detalle que el default 640

# --------------------------------------------------------------------------
# Lógica de pertenencia a la ROI
# --------------------------------------------------------------------------
# Punto del bounding box que se evalúa contra el polígono.
#   "centroid" -> centro geométrico (adecuado en vista cenital/nadir)
#   "bottom"   -> punto medio del borde inferior (adecuado en vista oblicua,
#                 aproxima el contacto del objeto con el suelo)
ANCHOR_MODE = "centroid"

# Histéresis temporal: número de frames consecutivos requeridos para confirmar
# una entrada o una salida. Evita que el parpadeo del detector en el borde del
# polígono infle artificialmente el número de visitas.
ENTER_CONSEC_FRAMES = 3
EXIT_CONSEC_FRAMES = 5

# Frames consecutivos sin observar un track antes de darlo por finalizado.
TRACK_TIMEOUT_FRAMES = 30

# --------------------------------------------------------------------------
# Codificación de video de salida
# --------------------------------------------------------------------------
# 'avc1' = H.264. Se intenta primero; si el build de OpenCV no lo soporta se
# cae a 'mp4v' (MPEG-4 Part 2), que produce archivos mucho más pesados.
FOURCC_PREFERRED = ("avc1", "mp4v")

# --------------------------------------------------------------------------
# ROI
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ROI:
    """Región de interés definida por un polígono en coordenadas normalizadas."""

    id: str
    name: str
    points_norm: tuple[tuple[float, float], ...]
    alert_seconds: float | None = None      # None = sin alerta de permanencia
    color: tuple[int, int, int] = (255, 0, 255)   # BGR

    def __post_init__(self) -> None:
        if len(self.points_norm) < 3:
            raise ValueError(f"ROI '{self.id}': un polígono requiere al menos 3 vértices.")
        for x, y in self.points_norm:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    f"ROI '{self.id}': vértice ({x}, {y}) fuera del rango normalizado [0, 1]."
                )

    def polygon(self, width: int, height: int) -> np.ndarray:
        """Devuelve el polígono en píxeles para un frame de (width, height)."""
        return np.array(
            [(int(round(x * width)), int(round(y * height))) for x, y in self.points_norm],
            dtype=np.int32,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["points_norm"] = [list(p) for p in self.points_norm]
        d["color"] = list(self.color)
        return d

    @staticmethod
    def from_dict(d: dict) -> "ROI":
        return ROI(
            id=d["id"],
            name=d["name"],
            points_norm=tuple(tuple(float(v) for v in p) for p in d["points_norm"]),
            alert_seconds=d.get("alert_seconds"),
            color=tuple(int(v) for v in d.get("color", (255, 0, 255))),
        )


# ADVERTENCIA: estas coordenadas son una PROPUESTA INICIAL derivada de una
# inspección visual del video Track.mp4 (vista cenital de un estacionamiento).
# DEBEN validarse con `python src/roi_editor.py --preview` o con la vista previa
# de la app de Streamlit antes de reportar resultados en el informe.
DEFAULT_ROIS: list[ROI] = [
    ROI(
        id="pasillo_norte",
        name="Pasillo de circulacion norte",
        points_norm=((0.02, 0.235), (0.98, 0.235), (0.98, 0.350), (0.02, 0.350)),
        alert_seconds=12.0,          # un vehículo detenido >12 s obstruye el pasillo
        color=(0, 200, 255),         # ámbar
    ),
    ROI(
        id="bahias_centro",
        name="Bahias de parqueo centrales",
        points_norm=((0.300, 0.360), (0.740, 0.360), (0.740, 0.870), (0.300, 0.870)),
        alert_seconds=None,          # zona de estacionamiento: permanecer es lo normal
        color=(255, 120, 0),         # azul
    ),
    ROI(
        id="pasillo_sur",
        name="Pasillo de circulacion sur",
        points_norm=((0.280, 0.880), (0.780, 0.880), (0.780, 0.975), (0.280, 0.975)),
        alert_seconds=12.0,
        color=(80, 220, 80),         # verde
    ),
]


def load_rois(path: Path | str = ROIS_FILE_DEFAULT) -> list[ROI]:
    """Carga las ROIs desde JSON. Si el archivo no existe, lo crea con los defaults."""
    path = Path(path)
    if not path.exists():
        save_rois(DEFAULT_ROIS, path)
        return list(DEFAULT_ROIS)

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    rois = [ROI.from_dict(d) for d in data["rois"]]
    ids = [r.id for r in rois]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Hay identificadores de ROI duplicados en {path}: {ids}")
    if not rois:
        raise ValueError(f"{path} no define ninguna ROI.")
    return rois


def save_rois(rois: Sequence[ROI], path: Path | str = ROIS_FILE_DEFAULT) -> Path:
    """Persiste las ROIs en JSON (coordenadas normalizadas)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comentario": (
            "Coordenadas normalizadas [0,1] respecto al ancho y alto del frame. "
            "alert_seconds = null desactiva la alerta de permanencia para esa ROI."
        ),
        "rois": [r.to_dict() for r in rois],
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def ascii_safe(text: str) -> str:
    """
    Translitera a ASCII.

    Las fuentes Hershey de OpenCV no tienen glifos para caracteres acentuados
    ni para 'ñ': cv2.putText los renderiza como símbolos incorrectos. Todo texto
    que se dibuje sobre el frame debe pasar por aquí.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")
