"""
Núcleo del sistema: detección (YOLO) + seguimiento (ByteTrack) + máquina de
estados de permanencia por ROI.

Separación de responsabilidades:
  - `ROIMonitor`  : lógica pura de pertenencia/permanencia. No depende de YOLO,
                    lo que permite probarla con detecciones sintéticas.
  - `Annotator`   : dibujo sobre el frame.
  - `process_video`: generador que une detección, monitoreo y anotación. Lo
                    consumen tanto run_pipeline.py (CLI) como app.py (Streamlit).

El generador cede el control frame a frame para que la interfaz pueda mostrar
progreso en vivo sin duplicar el pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import cv2
import numpy as np

from config import (
    ANCHOR_MODE,
    COCO_NAMES,
    CONF_THRES,
    ENTER_CONSEC_FRAMES,
    EXIT_CONSEC_FRAMES,
    FOURCC_PREFERRED,
    IMGSZ,
    MODEL_NAME,
    ROI,
    TARGET_CLASSES,
    TRACKER_CFG,
    TRACK_TIMEOUT_FRAMES,
    ascii_safe,
)

# --------------------------------------------------------------------------
# Estructuras de estado
# --------------------------------------------------------------------------


@dataclass
class Detection:
    """Una detección rastreada en un frame."""

    track_id: int
    cls_id: int
    conf: float
    xyxy: tuple[float, float, float, float]

    @property
    def cls_name(self) -> str:
        return COCO_NAMES.get(self.cls_id, str(self.cls_id))

    def anchor(self, mode: str = ANCHOR_MODE) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        if mode == "bottom":
            return ((x1 + x2) / 2.0, y2)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class ROIState:
    """Estado de permanencia de un track concreto respecto a una ROI concreta."""

    inside: bool = False
    visits: int = 0
    frames_inside: int = 0          # acumulado sobre todas las visitas
    current_visit_frames: int = 0
    max_visit_frames: int = 0
    first_entry_frame: int | None = None
    last_exit_frame: int | None = None
    alerts_raised: int = 0
    _alert_active: bool = False
    _in_streak: int = 0
    _out_streak: int = 0
    # Frames contabilizados como "dentro" mientras el objeto ya estaba
    # geometricamente fuera, a la espera de confirmar la salida. Se descuentan
    # al cerrar la visita para que la histeresis no sesgue el tiempo medido.
    _counted_while_out: int = 0


@dataclass
class TrackRecord:
    """Historial completo de un identificador de seguimiento."""

    track_id: int
    cls_id: int
    first_frame: int
    last_frame: int
    frames_seen: int = 0
    conf_sum: float = 0.0
    roi_states: dict[str, ROIState] = field(default_factory=dict)

    @property
    def cls_name(self) -> str:
        return COCO_NAMES.get(self.cls_id, str(self.cls_id))

    @property
    def conf_mean(self) -> float:
        return self.conf_sum / self.frames_seen if self.frames_seen else 0.0


@dataclass
class Alert:
    """Alerta emitida cuando un objeto excede el tiempo permitido en una ROI."""

    frame_idx: int
    timestamp_s: float
    track_id: int
    cls_name: str
    roi_id: str
    roi_name: str
    dwell_s: float
    threshold_s: float


@dataclass
class FrameSnapshot:
    """Fotografía del estado del sistema en un frame."""

    frame_idx: int
    timestamp_s: float
    detections: list[Detection]
    # roi_id -> lista de track_ids confirmados dentro en este frame
    occupancy: dict[str, list[int]]
    # track_id -> {roi_id: segundos de la visita en curso}
    dwell_now: dict[int, dict[str, float]]
    new_alerts: list[Alert]
    active_alerts: list[Alert]


# --------------------------------------------------------------------------
# Monitor de ROIs
# --------------------------------------------------------------------------


class ROIMonitor:
    """
    Máquina de estados que decide, para cada par (track, ROI), si el objeto está
    dentro, cuántas visitas ha hecho y cuánto tiempo acumula.

    La confirmación de entrada/salida usa histéresis: se requieren
    `enter_frames` observaciones consecutivas dentro para confirmar una entrada y
    `exit_frames` fuera para confirmar la salida. Sin esta histéresis, una caja
    que oscila sobre el borde del polígono genera decenas de visitas espurias y
    corrompe tanto el conteo como el tiempo promedio de permanencia.
    """

    def __init__(
        self,
        rois: Sequence[ROI],
        frame_size: tuple[int, int],
        fps: float,
        anchor_mode: str = ANCHOR_MODE,
        enter_frames: int = ENTER_CONSEC_FRAMES,
        exit_frames: int = EXIT_CONSEC_FRAMES,
        track_timeout: int = TRACK_TIMEOUT_FRAMES,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps debe ser positivo para convertir frames a segundos.")
        self.rois = list(rois)
        self.width, self.height = frame_size
        self.fps = float(fps)
        self.anchor_mode = anchor_mode
        self.enter_frames = max(1, int(enter_frames))
        self.exit_frames = max(1, int(exit_frames))
        self.track_timeout = int(track_timeout)

        self._polygons = {r.id: r.polygon(self.width, self.height) for r in self.rois}
        self._roi_by_id = {r.id: r for r in self.rois}

        self.records: dict[int, TrackRecord] = {}
        self.alerts: list[Alert] = []
        self.max_occupancy: dict[str, int] = {r.id: 0 for r in self.rois}
        self._last_seen: dict[int, int] = {}

    # -- utilidades ------------------------------------------------------

    def frames_to_seconds(self, frames: int) -> float:
        return frames / self.fps

    def _contains(self, roi_id: str, point: tuple[float, float]) -> bool:
        # measureDist=False -> devuelve +1 dentro, 0 sobre el borde, -1 fuera.
        return cv2.pointPolygonTest(self._polygons[roi_id], (float(point[0]), float(point[1])), False) >= 0

    # -- actualización ---------------------------------------------------

    def update(self, frame_idx: int, detections: Iterable[Detection]) -> FrameSnapshot:
        detections = list(detections)
        timestamp = self.frames_to_seconds(frame_idx)

        occupancy: dict[str, list[int]] = {r.id: [] for r in self.rois}
        dwell_now: dict[int, dict[str, float]] = {}
        new_alerts: list[Alert] = []
        seen_ids: set[int] = set()

        for det in detections:
            seen_ids.add(det.track_id)
            rec = self.records.get(det.track_id)
            if rec is None:
                rec = TrackRecord(
                    track_id=det.track_id,
                    cls_id=det.cls_id,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    roi_states={r.id: ROIState() for r in self.rois},
                )
                self.records[det.track_id] = rec

            rec.last_frame = frame_idx
            rec.frames_seen += 1
            rec.conf_sum += det.conf
            self._last_seen[det.track_id] = frame_idx

            point = det.anchor(self.anchor_mode)
            dwell_now[det.track_id] = {}

            for roi in self.rois:
                st = rec.roi_states[roi.id]
                raw_inside = self._contains(roi.id, point)
                contabilizado = self._step(st, raw_inside, frame_idx)

                if contabilizado:
                    occupancy[roi.id].append(det.track_id)

                    dwell_s = self.frames_to_seconds(st.current_visit_frames)
                    dwell_now[det.track_id][roi.id] = dwell_s

                    if (
                        roi.alert_seconds is not None
                        and dwell_s >= roi.alert_seconds
                        and not st._alert_active
                    ):
                        st._alert_active = True
                        st.alerts_raised += 1
                        alert = Alert(
                            frame_idx=frame_idx,
                            timestamp_s=timestamp,
                            track_id=det.track_id,
                            cls_name=det.cls_name,
                            roi_id=roi.id,
                            roi_name=roi.name,
                            dwell_s=dwell_s,
                            threshold_s=roi.alert_seconds,
                        )
                        new_alerts.append(alert)
                        self.alerts.append(alert)

        # Tracks no observados en este frame: cuentan como "fuera" para que sus
        # visitas se cierren cuando el objeto abandona el encuadre.
        for track_id, rec in self.records.items():
            if track_id in seen_ids:
                continue
            for roi in self.rois:
                self._step(rec.roi_states[roi.id], False, frame_idx)

        for roi_id, ids in occupancy.items():
            self.max_occupancy[roi_id] = max(self.max_occupancy[roi_id], len(ids))

        active_alerts = [
            a
            for a in self.alerts
            if self.records[a.track_id].roi_states[a.roi_id]._alert_active
        ]

        return FrameSnapshot(
            frame_idx=frame_idx,
            timestamp_s=timestamp,
            detections=detections,
            occupancy=occupancy,
            dwell_now=dwell_now,
            new_alerts=new_alerts,
            active_alerts=active_alerts,
        )

    def _step(self, st: ROIState, raw_inside: bool, frame_idx: int) -> bool:
        """
        Avanza un frame la máquina de estados de un par (track, ROI).

        Devuelve True si este frame se contabiliza como permanencia dentro.

        La histéresis introduce dos sesgos opuestos que aquí se compensan de
        forma explícita:
          - Entrada: la confirmación llega `enter_frames` frames tarde, así que
            al confirmar se acreditan retroactivamente los `enter_frames - 1`
            frames en que el objeto ya estaba dentro.
          - Salida: mientras se espera la confirmación se han contabilizado
            frames en que el objeto ya estaba fuera; al cerrar la visita se
            descuentan (`_counted_while_out`).
        Sin esta compensación el tiempo de permanencia queda inflado en
        `exit_frames - enter_frames` frames por visita, un error sistemático
        que invalidaría el tiempo promedio reportado.
        """
        if raw_inside:
            st._in_streak += 1
            st._out_streak = 0
            if not st.inside and st._in_streak >= self.enter_frames:
                st.inside = True
                st.visits += 1
                credito = self.enter_frames - 1
                st.current_visit_frames = credito
                st.frames_inside += credito
                st._counted_while_out = 0
                st._alert_active = False
                if st.first_entry_frame is None:
                    st.first_entry_frame = max(0, frame_idx - credito)
        else:
            st._out_streak += 1
            st._in_streak = 0
            if st.inside and st._out_streak >= self.exit_frames:
                self._close_visit(st, first_out_frame=frame_idx - st._out_streak + 1)
                return False

        if st.inside:
            st.frames_inside += 1
            st.current_visit_frames += 1
            st._counted_while_out = 0 if raw_inside else st._counted_while_out + 1
            return True
        return False

    def _close_visit(self, st: ROIState, first_out_frame: int | None = None) -> None:
        st.frames_inside -= st._counted_while_out
        st.current_visit_frames -= st._counted_while_out
        st.max_visit_frames = max(st.max_visit_frames, st.current_visit_frames)
        st.inside = False
        if first_out_frame is not None:
            st.last_exit_frame = max(0, first_out_frame)
        st.current_visit_frames = 0
        st._counted_while_out = 0
        st._alert_active = False

    def finalize(self) -> None:
        """Cierra las visitas que seguían abiertas al terminar el video."""
        for rec in self.records.values():
            for st in rec.roi_states.values():
                if st.inside:
                    self._close_visit(st)


# --------------------------------------------------------------------------
# Anotación
# --------------------------------------------------------------------------


class Annotator:
    """Dibuja ROIs, bounding boxes, IDs, tiempos de permanencia y contadores."""

    PANEL_BG = (30, 30, 30)
    TEXT = (245, 245, 245)
    ALERT = (0, 0, 235)

    def __init__(self, rois: Sequence[ROI], frame_size: tuple[int, int]):
        self.rois = list(rois)
        self.width, self.height = frame_size
        self._polys = {r.id: r.polygon(self.width, self.height) for r in self.rois}
        # Escala tipográfica proporcional a la resolución para que el texto sea
        # legible tanto en 720p como en 4K.
        self.scale = max(0.5, self.width / 1920.0)
        self.thick = max(1, int(round(2 * self.scale)))

    # -- primitivas ------------------------------------------------------

    def _text(self, img, txt, org, scale_mult=1.0, color=None, thick_mult=1.0):
        cv2.putText(
            img,
            ascii_safe(txt),
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * self.scale * scale_mult,
            color or self.TEXT,
            max(1, int(self.thick * thick_mult)),
            cv2.LINE_AA,
        )

    def _panel(self, img, x, y, w, h, alpha=0.55):
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.PANEL_BG, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # -- capas -----------------------------------------------------------

    def draw_rois(self, img, snapshot: FrameSnapshot | None = None, monitor: ROIMonitor | None = None):
        overlay = img.copy()
        for roi in self.rois:
            cv2.fillPoly(overlay, [self._polys[roi.id]], roi.color)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

        for roi in self.rois:
            cv2.polylines(img, [self._polys[roi.id]], True, roi.color, self.thick + 1, cv2.LINE_AA)
            x, y = self._polys[roi.id][0]
            label = roi.name
            if snapshot is not None and monitor is not None:
                dentro = len(snapshot.occupancy.get(roi.id, []))
                unicos = sum(
                    1 for r in monitor.records.values() if r.roi_states[roi.id].visits > 0
                )
                label = f"{roi.name} | dentro: {dentro} | unicos: {unicos}"
            ty = max(int(24 * self.scale), int(y) - int(10 * self.scale))
            self._text(img, label, (int(x) + 6, ty), color=roi.color, thick_mult=1.2)
        return img

    def draw_detections(self, img, snapshot: FrameSnapshot):
        alerted = {(a.track_id, a.roi_id) for a in snapshot.active_alerts}
        for det in snapshot.detections:
            x1, y1, x2, y2 = (int(v) for v in det.xyxy)
            dwell = snapshot.dwell_now.get(det.track_id, {})
            in_alert = any((det.track_id, rid) in alerted for rid in dwell)

            if in_alert:
                color = self.ALERT
            elif dwell:
                # color de la primera ROI que lo contiene
                first_roi = next(iter(dwell))
                color = next(r.color for r in self.rois if r.id == first_roi)
            else:
                color = (200, 200, 200)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, self.thick)

            label = f"#{det.track_id} {det.cls_name}"
            if dwell:
                label += " " + " ".join(f"{v:.1f}s" for v in dwell.values())

            (tw, th), _ = cv2.getTextSize(
                ascii_safe(label), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.scale, self.thick
            )
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
            self._text(img, label, (x1 + 4, y1 - 5), color=(15, 15, 15))
            if in_alert:
                cv2.circle(img, (x2, y1), int(7 * self.scale), self.ALERT, -1)
        return img

    def draw_hud(self, img, snapshot: FrameSnapshot, monitor: ROIMonitor, proc_fps: float | None = None):
        lines = [
            f"Frame {snapshot.frame_idx}  t={snapshot.timestamp_s:.2f}s",
            f"Objetos unicos rastreados: {len(monitor.records)}",
        ]
        for roi in self.rois:
            dentro = len(snapshot.occupancy.get(roi.id, []))
            unicos = sum(1 for r in monitor.records.values() if r.roi_states[roi.id].visits > 0)
            visitas = sum(r.roi_states[roi.id].visits for r in monitor.records.values())
            lines.append(f"{roi.name}: dentro {dentro} | unicos {unicos} | visitas {visitas}")
        if proc_fps is not None:
            lines.append(f"Procesamiento: {proc_fps:.1f} FPS")

        pad = int(12 * self.scale)
        lh = int(30 * self.scale)
        w = int(max(len(l) for l in lines) * 12 * self.scale) + 2 * pad
        h = lh * len(lines) + pad
        self._panel(img, pad, pad, min(w, self.width - 2 * pad), h)
        for i, line in enumerate(lines):
            self._text(img, line, (2 * pad, pad + lh * (i + 1) - int(8 * self.scale)))

        if snapshot.active_alerts:
            banner_h = int(46 * self.scale)
            self._panel(img, pad, self.height - banner_h - pad, self.width - 2 * pad, banner_h, alpha=0.7)
            txts = [
                f"ALERTA ID {a.track_id} ({a.cls_name}) en {a.roi_name}"
                for a in snapshot.active_alerts[:3]
            ]
            extra = len(snapshot.active_alerts) - 3
            if extra > 0:
                txts.append(f"+{extra} mas")
            self._text(
                img,
                " | ".join(txts),
                (2 * pad, self.height - pad - int(banner_h / 3)),
                color=(80, 80, 255),
                thick_mult=1.3,
            )
        return img

    def render(self, frame, snapshot: FrameSnapshot, monitor: ROIMonitor, proc_fps: float | None = None):
        img = frame.copy()
        self.draw_rois(img, snapshot, monitor)
        self.draw_detections(img, snapshot)
        self.draw_hud(img, snapshot, monitor, proc_fps)
        return img


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def open_writer(path: Path | str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Abre un VideoWriter probando los códecs en orden de preferencia."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for code in FOURCC_PREFERRED:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        f"No se pudo abrir un VideoWriter para {path} con ninguno de los codecs {FOURCC_PREFERRED}."
    )


def _parse_results(result) -> list[Detection]:
    """Convierte el objeto Results de Ultralytics en una lista de Detection."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        # Sin IDs asignados en este frame (p. ej. no hubo detecciones).
        return []
    ids = boxes.id.int().cpu().tolist()
    clss = boxes.cls.int().cpu().tolist()
    confs = boxes.conf.float().cpu().tolist()
    xyxys = boxes.xyxy.float().cpu().tolist()
    return [
        Detection(track_id=int(i), cls_id=int(c), conf=float(cf), xyxy=tuple(float(v) for v in xy))
        for i, c, cf, xy in zip(ids, clss, confs, xyxys)
    ]


def process_video(
    video_in: Path | str,
    rois: Sequence[ROI],
    video_out: Path | str | None = None,
    model_name: str = MODEL_NAME,
    classes: Sequence[int] = TARGET_CLASSES,
    conf: float = CONF_THRES,
    imgsz: int = IMGSZ,
    anchor_mode: str = ANCHOR_MODE,
    enter_frames: int = ENTER_CONSEC_FRAMES,
    exit_frames: int = EXIT_CONSEC_FRAMES,
    max_frames: int | None = None,
) -> Iterator[tuple[np.ndarray, FrameSnapshot, ROIMonitor, dict]]:
    """
    Generador que procesa el video y cede (frame_anotado, snapshot, monitor, meta).

    `meta` contiene progreso y rendimiento: total_frames, fps_video, proc_fps,
    elapsed_s. Al agotarse el generador, `monitor` queda finalizado y contiene el
    historial completo listo para stats.py.
    """
    from ultralytics import YOLO  # import diferido: permite testear sin ultralytics

    video_in = Path(video_in)
    if not video_in.exists():
        raise FileNotFoundError(f"No existe el video de entrada: {video_in}")

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV no pudo abrir el video: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    monitor = ROIMonitor(
        rois,
        frame_size=(width, height),
        fps=fps,
        anchor_mode=anchor_mode,
        enter_frames=enter_frames,
        exit_frames=exit_frames,
    )
    annotator = Annotator(rois, frame_size=(width, height))
    model = YOLO(model_name)

    writer = open_writer(video_out, fps, (width, height)) if video_out else None

    frame_idx = 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            results = model.track(
                frame,
                persist=True,
                classes=list(classes),
                conf=conf,
                imgsz=imgsz,
                max_det=1000,          # <-- agregar: la escena tiene >150 vehiculos
                tracker=TRACKER_CFG,
                verbose=False,
            )
            detections = _parse_results(results[0])
            snapshot = monitor.update(frame_idx, detections)

            elapsed = time.perf_counter() - t0
            proc_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
            annotated = annotator.render(frame, snapshot, monitor, proc_fps)

            if writer is not None:
                writer.write(annotated)

            yield annotated, snapshot, monitor, {
                "frame_idx": frame_idx,
                "total_frames": total_frames,
                "fps_video": fps,
                "proc_fps": proc_fps,
                "elapsed_s": elapsed,
                "width": width,
                "height": height,
            }
            frame_idx += 1
    finally:
        monitor.finalize()
        cap.release()
        if writer is not None:
            writer.release()
