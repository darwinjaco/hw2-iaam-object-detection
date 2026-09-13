"""
Diagnostico 2: objetos pequenos en vista cenital.

Compara dos estrategias contra el mismo frame:
  A) subir imgsz (reescalado global)
  B) inferencia por mosaicos (tiles solapados a escala nativa + fusion por NMS)

    python src/diagnostico2.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

CONF = 0.15


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> list[int]:
    """NMS clasica en numpy. boxes en formato xyxy."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def detectar_por_mosaicos(model, frame, filas: int, cols: int, solape: float = 0.2,
                          imgsz: int = 1280, conf: float = CONF):
    """Divide el frame en tiles solapados, detecta en cada uno y fusiona con NMS."""
    h, w = frame.shape[:2]
    th, tw = h // filas, w // cols
    dy, dx = int(th * solape), int(tw * solape)

    cajas, scores, clases = [], [], []
    for r in range(filas):
        for c in range(cols):
            y0 = max(0, r * th - dy)
            y1 = min(h, (r + 1) * th + dy)
            x0 = max(0, c * tw - dx)
            x1 = min(w, (c + 1) * tw + dx)
            tile = frame[y0:y1, x0:x1]

            res = model.predict(tile, classes=cfg.TARGET_CLASSES, conf=conf,
                                imgsz=imgsz, max_det=1000, verbose=False)
            b = res[0].boxes
            if b is None or len(b) == 0:
                continue
            xy = b.xyxy.cpu().numpy()
            xy[:, [0, 2]] += x0
            xy[:, [1, 3]] += y0
            cajas.append(xy)
            scores.append(b.conf.cpu().numpy())
            clases.append(b.cls.cpu().numpy())

    if not cajas:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
    cajas = np.vstack(cajas)
    scores = np.concatenate(scores)
    clases = np.concatenate(clases)
    keep = nms(cajas, scores, 0.5)
    return cajas[keep], scores[keep], clases[keep]


def dibujar(frame, cajas, titulo, ruta):
    img = frame.copy()
    for x1, y1, x2, y2 in cajas.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(img, (0, 0), (900, 60), (20, 20, 20), -1)
    cv2.putText(img, f"{titulo}: {len(cajas)} detecciones", (16, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(ruta), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> int:
    from ultralytics import YOLO

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(cfg.VIDEO_IN_DEFAULT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("No se pudo leer el frame.")
        return 1

    modelo = "yolo26l.pt"
    m = YOLO(modelo)

    print(f"\nA) Reescalado global  (modelo {modelo}, conf {CONF})")
    print(f"{'imgsz':>8}{'DETECT':>9}{'ms/frame':>11}")
    print("-" * 28)
    mejor_global = (0, None, 0)
    for imgsz in (1920, 2560, 3200, 4096):
        try:
            t0 = time.perf_counter()
            r = m.predict(frame, classes=cfg.TARGET_CLASSES, conf=CONF,
                          imgsz=imgsz, max_det=1000, verbose=False)
            ms = (time.perf_counter() - t0) * 1000
            b = r[0].boxes
            n = 0 if b is None else len(b)
            print(f"{imgsz:>8}{n:>9}{ms:>11.0f}")
            if n > mejor_global[0]:
                mejor_global = (n, b.xyxy.cpu().numpy() if n else None, imgsz)
        except Exception as exc:  # noqa: BLE001
            print(f"{imgsz:>8}   fallo: {type(exc).__name__}")

    print(f"\nB) Inferencia por mosaicos  (tiles a imgsz 1280, solape 20%)")
    print(f"{'grilla':>8}{'DETECT':>9}{'ms/frame':>11}")
    print("-" * 28)
    mejor_tile = (0, None, "")
    for filas, cols in ((2, 2), (2, 3), (3, 4), (4, 6)):
        t0 = time.perf_counter()
        cajas, _s, _c = detectar_por_mosaicos(m, frame, filas, cols)
        ms = (time.perf_counter() - t0) * 1000
        print(f"{f'{filas}x{cols}':>8}{len(cajas):>9}{ms:>11.0f}")
        if len(cajas) > mejor_tile[0]:
            mejor_tile = (len(cajas), cajas, f"{filas}x{cols}")

    if mejor_global[1] is not None:
        dibujar(frame, mejor_global[1], f"Global imgsz {mejor_global[2]}",
                cfg.OUTPUT_DIR / "diag2_global.jpg")
    if mejor_tile[1] is not None:
        dibujar(frame, mejor_tile[1], f"Mosaicos {mejor_tile[2]}",
                cfg.OUTPUT_DIR / "diag2_mosaicos.jpg")

    print("\nMejor global : "
          f"{mejor_global[0]} detecciones (imgsz {mejor_global[2]})")
    print("Mejor mosaico: "
          f"{mejor_tile[0]} detecciones (grilla {mejor_tile[2]})")
    print("\nCompara outputs/diag2_global.jpg y outputs/diag2_mosaicos.jpg.")
    print("Mira si las cajas caen sobre autos reales o si hay falsos positivos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
