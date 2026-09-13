"""
Definición y verificación de las ROIs sobre un frame real del video.

Existe para eliminar el supuesto no verificado: las coordenadas de una ROI no
deben "estimarse a ojo" en el código, sino trazarse sobre el video y quedar
documentadas en config/rois.json.

Modos
-----
    python src/roi_editor.py --preview
        Genera outputs/roi_preview.jpg con las ROIs actuales dibujadas sobre un
        frame del video. No requiere entorno grafico. Usar SIEMPRE antes de
        reportar resultados.

    python src/roi_editor.py
        Editor interactivo (requiere OpenCV con soporte GUI).
        Click izquierdo : agregar vertice
        u               : deshacer ultimo vertice
        n               : cerrar la ROI actual y comenzar otra
        r               : reiniciar todo
        s               : guardar en config/rois.json y salir
        q / ESC         : salir sin guardar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg

PALETA = [(0, 200, 255), (255, 120, 0), (80, 220, 80), (200, 80, 255), (0, 255, 255)]


def leer_frame(video: Path, indice: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total and indice >= total:
        indice = max(0, total // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"No se pudo leer el frame {indice} de {video}")
    return frame


def dibujar_rois(frame: np.ndarray, rois, con_rejilla: bool = False) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    escala = max(0.5, w / 1920.0)

    if con_rejilla:
        for i in range(1, 10):
            x, y = int(w * i / 10), int(h * i / 10)
            cv2.line(img, (x, 0), (x, h), (60, 60, 60), 1)
            cv2.line(img, (0, y), (w, y), (60, 60, 60), 1)
            cv2.putText(img, f"{i/10:.1f}", (x + 4, int(24 * escala)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * escala, (60, 60, 60), 1, cv2.LINE_AA)
            cv2.putText(img, f"{i/10:.1f}", (4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * escala, (60, 60, 60), 1, cv2.LINE_AA)

    overlay = img.copy()
    for roi in rois:
        cv2.fillPoly(overlay, [roi.polygon(w, h)], roi.color)
    cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)

    for roi in rois:
        poly = roi.polygon(w, h)
        cv2.polylines(img, [poly], True, roi.color, max(2, int(3 * escala)), cv2.LINE_AA)
        for px, py in poly:
            cv2.circle(img, (int(px), int(py)), max(3, int(5 * escala)), roi.color, -1)
        etiqueta = cfg.ascii_safe(f"{roi.id}: {roi.name}")
        if roi.alert_seconds is not None:
            etiqueta += f" (alerta {roi.alert_seconds:g}s)"
        x, y = poly[0]
        cv2.putText(img, etiqueta, (int(x) + 6, max(int(26 * escala), int(y) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65 * escala, roi.color,
                    max(1, int(2 * escala)), cv2.LINE_AA)
    return img


def modo_preview(args) -> int:
    rois = cfg.load_rois(args.rois)
    frame = leer_frame(args.input, args.frame)
    img = dibujar_rois(frame, rois, con_rejilla=args.rejilla)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.salida), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Vista previa -> {args.salida}")
    print(f"Frame {args.frame} de {args.input} | resolucion {frame.shape[1]}x{frame.shape[0]}")
    print("\nROIs cargadas:")
    for r in rois:
        pts = ", ".join(f"({x:.3f},{y:.3f})" for x, y in r.points_norm)
        print(f"  {r.id:<16} alerta={r.alert_seconds}  vertices=[{pts}]")
    print("\nVerifica visualmente que cada poligono cubra la zona esperada.")
    print("Si no coincide, ajusta config/rois.json o usa el editor interactivo.")
    return 0


def modo_interactivo(args) -> int:
    frame = leer_frame(args.input, args.frame)
    h, w = frame.shape[:2]
    poligonos: list[list[tuple[int, int]]] = []
    actual: list[tuple[int, int]] = []

    escala_vista = min(1.0, 1280 / w)
    ventana = "Editor de ROIs  |  click: vertice  u: deshacer  n: nueva ROI  r: reset  s: guardar  q: salir"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            actual.append((int(x / escala_vista), int(y / escala_vista)))

    try:
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    except cv2.error:
        print("Este build de OpenCV no tiene soporte GUI. Usa --preview y edita "
              "config/rois.json a mano, o ejecuta la app de Streamlit.", file=sys.stderr)
        return 2
    cv2.resizeWindow(ventana, int(w * escala_vista), int(h * escala_vista))
    cv2.setMouseCallback(ventana, on_mouse)

    while True:
        lienzo = frame.copy()
        for i, poly in enumerate(poligonos):
            color = PALETA[i % len(PALETA)]
            cv2.polylines(lienzo, [np.array(poly, np.int32)], True, color, 3, cv2.LINE_AA)
        if actual:
            color = PALETA[len(poligonos) % len(PALETA)]
            cv2.polylines(lienzo, [np.array(actual, np.int32)], False, color, 3, cv2.LINE_AA)
            for p in actual:
                cv2.circle(lienzo, p, 6, color, -1)
        cv2.putText(lienzo, f"ROIs cerradas: {len(poligonos)} | vertices actuales: {len(actual)}",
                    (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(ventana, cv2.resize(lienzo, None, fx=escala_vista, fy=escala_vista))
        k = cv2.waitKey(20) & 0xFF

        if k in (ord("q"), 27):
            cv2.destroyAllWindows()
            print("Salida sin guardar.")
            return 0
        if k == ord("u") and actual:
            actual.pop()
        elif k == ord("r"):
            poligonos.clear()
            actual.clear()
        elif k == ord("n"):
            if len(actual) >= 3:
                poligonos.append(list(actual))
                actual.clear()
            else:
                print("Una ROI necesita al menos 3 vertices.")
        elif k == ord("s"):
            if len(actual) >= 3:
                poligonos.append(list(actual))
                actual.clear()
            if not poligonos:
                print("No hay ninguna ROI que guardar.")
                continue
            cv2.destroyAllWindows()
            rois = []
            for i, poly in enumerate(poligonos):
                rid = input(f"ID de la ROI {i+1} (sin espacios): ").strip() or f"roi_{i+1}"
                nombre = input(f"Nombre descriptivo de '{rid}': ").strip() or rid
                umbral = input(f"Umbral de alerta en segundos para '{rid}' (Enter = sin alerta): ").strip()
                rois.append(
                    cfg.ROI(
                        id=rid,
                        name=nombre,
                        points_norm=tuple((x / w, y / h) for x, y in poly),
                        alert_seconds=float(umbral) if umbral else None,
                        color=PALETA[i % len(PALETA)],
                    )
                )
            ruta = cfg.save_rois(rois, args.rois)
            print(f"Guardado -> {ruta}")
            return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Editor y verificador de ROIs.")
    p.add_argument("--input", type=Path, default=cfg.VIDEO_IN_DEFAULT)
    p.add_argument("--rois", type=Path, default=cfg.ROIS_FILE_DEFAULT)
    p.add_argument("--frame", type=int, default=200, help="Indice del frame de referencia.")
    p.add_argument("--preview", action="store_true", help="Solo genera la imagen de verificacion.")
    p.add_argument("--rejilla", action="store_true", help="Superpone una rejilla normalizada.")
    p.add_argument("--salida", type=Path, default=cfg.OUTPUT_DIR / "roi_preview.jpg")
    args = p.parse_args()
    return modo_preview(args) if args.preview else modo_interactivo(args)


if __name__ == "__main__":
    raise SystemExit(main())
