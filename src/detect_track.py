"""
Objetivo 1 (linea base): deteccion YOLO + seguimiento ByteTrack, sin ROI.

Sirve para verificar que el detector encuentra los vehiculos y que ByteTrack
mantiene identificadores estables antes de construir la logica de ROI encima.
Tambien mide los FPS de procesamiento, dato necesario para sustentar cualquier
afirmacion sobre operacion en tiempo real.

    python src/detect_track.py --max-frames 150
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from roi_tracker import open_writer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linea base: deteccion + tracking.")
    p.add_argument("--input", type=Path, default=cfg.VIDEO_IN_DEFAULT)
    p.add_argument("--output", type=Path, default=cfg.VIDEO_OUT_BASE)
    p.add_argument("--model", default=cfg.MODEL_NAME)
    p.add_argument("--clases", type=int, nargs="+", default=cfg.TARGET_CLASSES)
    p.add_argument("--conf", type=float, default=cfg.CONF_THRES)
    p.add_argument("--imgsz", type=int, default=cfg.IMGSZ)
    p.add_argument("--max-frames", type=int, default=None)
    return p.parse_args()


def main() -> int:
    from ultralytics import YOLO

    args = parse_args()
    if not args.input.exists():
        print(f"ERROR: no existe el video {args.input}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"ERROR: OpenCV no pudo abrir {args.input}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    model = YOLO(args.model)
    writer = open_writer(args.output, fps, (w, h))

    ids_vistos: set[int] = set()
    n = 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames is not None and n >= args.max_frames:
                break

            results = model.track(
                frame,
                persist=True,                 # conserva los IDs entre frames
                classes=args.clases,
                conf=args.conf,
                imgsz=args.imgsz,
                tracker=cfg.TRACKER_CFG,      # ByteTrack, requisito de la tarea
                verbose=False,
            )
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                ids_vistos.update(int(i) for i in boxes.id.int().cpu().tolist())

            writer.write(results[0].plot())   # dibuja boxes + ID de tracking
            n += 1
            if n % 30 == 0:
                print(f"  frame {n}/{total or '?'}", end="\r", flush=True)
    finally:
        cap.release()
        writer.release()

    dt = time.perf_counter() - t0
    print(f"\n{n} frames -> {args.output}")
    print(f"Tiempo total: {dt:.1f}s | Procesamiento: {n/dt:.2f} FPS | Video: {fps:.2f} FPS")
    print(f"Ratio tiempo real: {(n/dt)/fps:.2f}x  (>=1.0 significa procesamiento en tiempo real)")
    print(f"Identificadores unicos asignados por ByteTrack: {len(ids_vistos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
