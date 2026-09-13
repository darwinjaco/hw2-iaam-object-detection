"""
Diagnostico: separa cuantos objetos DETECTA el modelo de cuantos RASTREA el tracker.

Si detect >> track, el cuello de botella son los umbrales de bytetrack.yaml.
Si detect tambien es bajo, el problema es el detector en vista cenital.

    python src/diagnostico.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def main() -> int:
    from ultralytics import YOLO

    try:
        import torch
        gpu = torch.cuda.is_available()
        print(f"CUDA disponible: {gpu}" + (f" | {torch.cuda.get_device_name(0)}" if gpu else ""))
        print(f"torch: {torch.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"No se pudo consultar torch: {exc}")

    cap = cv2.VideoCapture(str(cfg.VIDEO_IN_DEFAULT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("No se pudo leer el frame de prueba.")
        return 1
    print(f"Frame de prueba: {frame.shape[1]}x{frame.shape[0]}\n")

    print(f"{'modelo':<12}{'imgsz':>7}{'conf':>7}{'DETECT':>9}{'TRACK':>8}")
    print("-" * 45)

    for modelo in ("yolo26l.pt", "yolo11l.pt"):
        try:
            m = YOLO(modelo)
        except Exception as exc:  # noqa: BLE001
            print(f"{modelo:<12}  no disponible ({type(exc).__name__})")
            continue

        for imgsz in (1280, 1600, 1920):
            for conf in (0.25, 0.10, 0.05):
                # Deteccion pura, sin tracker
                r = m.predict(frame, classes=cfg.TARGET_CLASSES, conf=conf,
                              imgsz=imgsz, max_det=1000, verbose=False)
                n_det = len(r[0].boxes) if r[0].boxes is not None else 0

                # Con tracker (primer frame: solo se crean tracks nuevos)
                m.predictor = None
                rt = m.track(frame, persist=False, classes=cfg.TARGET_CLASSES,
                             conf=conf, imgsz=imgsz, max_det=1000,
                             tracker=cfg.TRACKER_CFG, verbose=False)
                b = rt[0].boxes
                n_trk = 0 if b is None or b.id is None else len(b.id)

                print(f"{modelo:<12}{imgsz:>7}{conf:>7.2f}{n_det:>9}{n_trk:>8}")

                if imgsz == 1600 and conf == 0.10:
                    cv2.imwrite(str(cfg.OUTPUT_DIR / f"diag_{modelo.split('.')[0]}.jpg"),
                                r[0].plot(), [cv2.IMWRITE_JPEG_QUALITY, 90])

    print("\nRevisa outputs/diag_*.jpg para ver que autos encuentra realmente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
