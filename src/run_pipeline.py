"""
Ejecución por línea de comandos del pipeline completo.

Genera el video anotado y todas las tablas de estadísticas.

Ejemplos
--------
    python src/run_pipeline.py
    python src/run_pipeline.py --clases 2 5 7 --conf 0.4 --imgsz 1280
    python src/run_pipeline.py --max-frames 150      # prueba rápida
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar el script desde cualquier directorio.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import stats
from roi_tracker import process_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deteccion + ByteTrack + monitoreo de permanencia en multiples ROIs."
    )
    p.add_argument("--input", type=Path, default=cfg.VIDEO_IN_DEFAULT, help="Video de entrada.")
    p.add_argument("--output", type=Path, default=cfg.VIDEO_OUT_DEFAULT, help="Video anotado de salida.")
    p.add_argument("--rois", type=Path, default=cfg.ROIS_FILE_DEFAULT, help="JSON con las ROIs.")
    p.add_argument("--outdir", type=Path, default=cfg.OUTPUT_DIR, help="Carpeta para CSV/XLSX.")
    p.add_argument("--model", default=cfg.MODEL_NAME, help="Pesos YOLO.")
    p.add_argument("--clases", type=int, nargs="+", default=cfg.TARGET_CLASSES,
                   help="IDs de clase COCO a detectar.")
    p.add_argument("--conf", type=float, default=cfg.CONF_THRES, help="Umbral de confianza.")
    p.add_argument("--imgsz", type=int, default=cfg.IMGSZ, help="Tamano de inferencia.")
    p.add_argument("--anchor", choices=["centroid", "bottom"], default=cfg.ANCHOR_MODE,
                   help="Punto del box evaluado contra el poligono.")
    p.add_argument("--enter-frames", type=int, default=cfg.ENTER_CONSEC_FRAMES)
    p.add_argument("--exit-frames", type=int, default=cfg.EXIT_CONSEC_FRAMES)
    p.add_argument("--max-frames", type=int, default=None, help="Limita el numero de frames (pruebas).")
    p.add_argument("--sin-video", action="store_true", help="Solo calcula estadisticas, no escribe video.")
    p.add_argument("--sin-excel", action="store_true", help="Exporta solo CSV.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rois = cfg.load_rois(args.rois)

    print(f"Video      : {args.input}")
    print(f"Modelo     : {args.model} | tracker: {cfg.TRACKER_CFG}")
    print(f"Clases     : {args.clases} -> {[cfg.COCO_NAMES.get(c, c) for c in args.clases]}")
    print(f"conf={args.conf}  imgsz={args.imgsz}  anchor={args.anchor}")
    print(f"ROIs ({len(rois)}): " + ", ".join(f"{r.id}" for r in rois))
    print("-" * 72)

    monitor = None
    meta: dict = {}
    gen = process_video(
        video_in=args.input,
        rois=rois,
        video_out=None if args.sin_video else args.output,
        model_name=args.model,
        classes=args.clases,
        conf=args.conf,
        imgsz=args.imgsz,
        anchor_mode=args.anchor,
        enter_frames=args.enter_frames,
        exit_frames=args.exit_frames,
        max_frames=args.max_frames,
    )

    for _frame, snapshot, monitor, meta in gen:
        if snapshot.frame_idx % 30 == 0:
            total = meta.get("total_frames") or "?"
            print(
                f"  frame {snapshot.frame_idx:>5}/{total} | "
                f"objetos {len(snapshot.detections):>3} | "
                f"{meta['proc_fps']:.1f} FPS",
                end="\r",
                flush=True,
            )
        for a in snapshot.new_alerts:
            print(f"\n  [ALERTA] t={a.timestamp_s:6.2f}s  ID {a.track_id} ({a.cls_name}) "
                  f"lleva {a.dwell_s:.1f}s en '{a.roi_name}' (umbral {a.threshold_s}s)")

    print()
    if monitor is None:
        print("ERROR: no se proceso ningun frame. Revisa el video de entrada.", file=sys.stderr)
        return 1

    if not args.sin_video:
        print(f"Video anotado -> {args.output}")

    escritos = stats.exportar(monitor, args.outdir, meta=meta, excel=not args.sin_excel)
    stats.imprimir_resumen(monitor, meta)
    print("Archivos generados:")
    for nombre, ruta in escritos.items():
        print(f"  - {nombre:<24} {ruta}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
