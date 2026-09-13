"""
Agregación de estadísticas y exportación a CSV / Excel.

Definiciones usadas (deben citarse tal cual en el informe para que los números
sean interpretables):

  - Objeto único: un `track_id` distinto asignado por ByteTrack. No equivale
    necesariamente a un vehículo físico distinto: una oclusión prolongada puede
    fragmentar un vehículo en dos IDs. Es una cota superior del conteo real.

  - Visita: cada transición confirmada fuera -> dentro de una ROI, aplicando la
    histéresis de `ENTER_CONSEC_FRAMES` / `EXIT_CONSEC_FRAMES`.

  - Tiempo de permanencia por objeto: suma de frames confirmados dentro de la
    ROI dividida entre los FPS del video, agregando todas sus visitas.

  - Tiempo promedio de permanencia: media del tiempo de permanencia por objeto,
    calculada solo sobre los objetos con al menos una visita a esa ROI. Se
    reporta además la media por visita, que es distinta cuando hay reingresos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from config import ROI
from roi_tracker import ROIMonitor


# --------------------------------------------------------------------------
# Construcción de tablas
# --------------------------------------------------------------------------


def tabla_permanencia(monitor: ROIMonitor) -> pd.DataFrame:
    """Formato largo: una fila por par (objeto, ROI) con al menos una visita."""
    filas = []
    for rec in monitor.records.values():
        for roi in monitor.rois:
            st = rec.roi_states[roi.id]
            if st.visits == 0:
                continue
            filas.append(
                {
                    "track_id": rec.track_id,
                    "clase": rec.cls_name,
                    "confianza_promedio": round(rec.conf_mean, 4),
                    "frame_primera_deteccion": rec.first_frame,
                    "frame_ultima_deteccion": rec.last_frame,
                    "tiempo_visible_s": round(monitor.frames_to_seconds(rec.frames_seen), 3),
                    "roi_id": roi.id,
                    "roi_nombre": roi.name,
                    "visitas": st.visits,
                    "frames_dentro": st.frames_inside,
                    "tiempo_en_roi_s": round(monitor.frames_to_seconds(st.frames_inside), 3),
                    "visita_mas_larga_s": round(monitor.frames_to_seconds(st.max_visit_frames), 3),
                    "frame_primera_entrada": st.first_entry_frame,
                    "frame_ultima_salida": st.last_exit_frame,
                    "alertas": st.alerts_raised,
                }
            )
    cols = [
        "track_id", "clase", "confianza_promedio", "frame_primera_deteccion",
        "frame_ultima_deteccion", "tiempo_visible_s", "roi_id", "roi_nombre",
        "visitas", "frames_dentro", "tiempo_en_roi_s", "visita_mas_larga_s",
        "frame_primera_entrada", "frame_ultima_salida", "alertas",
    ]
    df = pd.DataFrame(filas, columns=cols)
    return df.sort_values(["roi_id", "track_id"]).reset_index(drop=True) if not df.empty else df


def tabla_objetos(monitor: ROIMonitor) -> pd.DataFrame:
    """Una fila por objeto rastreado, haya entrado o no a alguna ROI."""
    filas = []
    for rec in monitor.records.values():
        fila = {
            "track_id": rec.track_id,
            "clase": rec.cls_name,
            "confianza_promedio": round(rec.conf_mean, 4),
            "frames_visible": rec.frames_seen,
            "tiempo_visible_s": round(monitor.frames_to_seconds(rec.frames_seen), 3),
            "rois_visitadas": sum(1 for st in rec.roi_states.values() if st.visits > 0),
        }
        for roi in monitor.rois:
            st = rec.roi_states[roi.id]
            fila[f"visitas__{roi.id}"] = st.visits
            fila[f"tiempo_s__{roi.id}"] = round(monitor.frames_to_seconds(st.frames_inside), 3)
        filas.append(fila)
    df = pd.DataFrame(filas)
    return df.sort_values("track_id").reset_index(drop=True) if not df.empty else df


def resumen_por_roi(monitor: ROIMonitor) -> pd.DataFrame:
    """Estadísticas finales por ROI: la tabla que pide el reto de 5 pts."""
    perm = tabla_permanencia(monitor)
    filas = []
    for roi in monitor.rois:
        sub = perm[perm["roi_id"] == roi.id] if not perm.empty else perm
        tiempos = sub["tiempo_en_roi_s"] if len(sub) else pd.Series(dtype="float64")
        visitas_totales = int(sub["visitas"].sum()) if len(sub) else 0
        frames_totales = int(sub["frames_dentro"].sum()) if len(sub) else 0
        filas.append(
            {
                "roi_id": roi.id,
                "roi_nombre": roi.name,
                "objetos_unicos": int(len(sub)),
                "visitas_totales": visitas_totales,
                "tiempo_promedio_por_objeto_s": round(float(tiempos.mean()), 3) if len(tiempos) else 0.0,
                "tiempo_promedio_por_visita_s": (
                    round(monitor.frames_to_seconds(frames_totales) / visitas_totales, 3)
                    if visitas_totales
                    else 0.0
                ),
                "tiempo_mediana_s": round(float(tiempos.median()), 3) if len(tiempos) else 0.0,
                "tiempo_maximo_s": round(float(tiempos.max()), 3) if len(tiempos) else 0.0,
                "ocupacion_maxima_simultanea": monitor.max_occupancy.get(roi.id, 0),
                "umbral_alerta_s": roi.alert_seconds,
                "alertas_emitidas": int(sub["alertas"].sum()) if len(sub) else 0,
            }
        )
    return pd.DataFrame(filas)


def tabla_alertas(monitor: ROIMonitor) -> pd.DataFrame:
    filas = [
        {
            "frame": a.frame_idx,
            "timestamp_s": round(a.timestamp_s, 3),
            "track_id": a.track_id,
            "clase": a.cls_name,
            "roi_id": a.roi_id,
            "roi_nombre": a.roi_name,
            "permanencia_s": round(a.dwell_s, 3),
            "umbral_s": a.threshold_s,
        }
        for a in monitor.alerts
    ]
    cols = ["frame", "timestamp_s", "track_id", "clase", "roi_id", "roi_nombre",
            "permanencia_s", "umbral_s"]
    return pd.DataFrame(filas, columns=cols)


def resumen_global(monitor: ROIMonitor, meta: dict | None = None) -> pd.DataFrame:
    """Pares métrica/valor con los indicadores globales del procesamiento."""
    perm = tabla_permanencia(monitor)
    objetos = tabla_objetos(monitor)

    conteo_clases = (
        objetos["clase"].value_counts().to_dict() if not objetos.empty else {}
    )
    objetos_en_alguna_roi = (
        int(perm["track_id"].nunique()) if not perm.empty else 0
    )
    tiempo_prom_global = (
        round(float(perm["tiempo_en_roi_s"].mean()), 3) if not perm.empty else 0.0
    )

    filas = [
        ("total_objetos_detectados", len(monitor.records)),
        ("objetos_que_entraron_a_alguna_roi", objetos_en_alguna_roi),
        ("visitas_totales_todas_las_rois", int(perm["visitas"].sum()) if not perm.empty else 0),
        ("tiempo_promedio_permanencia_global_s", tiempo_prom_global),
        ("alertas_totales", len(monitor.alerts)),
        ("numero_de_rois", len(monitor.rois)),
        ("fps_video", round(monitor.fps, 3)),
    ]
    for clase, n in sorted(conteo_clases.items()):
        filas.append((f"objetos_clase_{clase}", int(n)))
    if meta:
        filas += [
            ("frames_procesados", meta.get("frame_idx", 0) + 1),
            ("resolucion", f"{meta.get('width')}x{meta.get('height')}"),
            ("fps_procesamiento", round(meta.get("proc_fps", 0.0), 2)),
            ("tiempo_total_procesamiento_s", round(meta.get("elapsed_s", 0.0), 2)),
        ]
    return pd.DataFrame(filas, columns=["metrica", "valor"])


# --------------------------------------------------------------------------
# Exportación
# --------------------------------------------------------------------------


def exportar(
    monitor: ROIMonitor,
    output_dir: Path | str,
    meta: dict | None = None,
    prefijo: str = "",
    excel: bool = True,
) -> dict[str, Path]:
    """Escribe todas las tablas a CSV (UTF-8 con BOM para Excel) y opcionalmente a XLSX."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tablas = {
        "resumen_global": resumen_global(monitor, meta),
        "resumen_por_roi": resumen_por_roi(monitor),
        "permanencia_por_objeto": tabla_permanencia(monitor),
        "objetos": tabla_objetos(monitor),
        "alertas": tabla_alertas(monitor),
    }

    escritos: dict[str, Path] = {}
    for nombre, df in tablas.items():
        ruta = output_dir / f"{prefijo}{nombre}.csv"
        # utf-8-sig evita que Excel muestre mal los acentos de los encabezados.
        df.to_csv(ruta, index=False, encoding="utf-8-sig")
        escritos[nombre] = ruta

    if excel:
        ruta_xlsx = output_dir / f"{prefijo}estadisticas.xlsx"
        try:
            with pd.ExcelWriter(ruta_xlsx, engine="openpyxl") as xw:
                for nombre, df in tablas.items():
                    df.to_excel(xw, sheet_name=nombre[:31], index=False)
            escritos["excel"] = ruta_xlsx
        except ImportError:
            # openpyxl no instalado: los CSV siguen siendo válidos.
            pass

    return escritos


def imprimir_resumen(monitor: ROIMonitor, meta: dict | None = None) -> None:
    """Resumen legible en consola."""
    print("\n" + "=" * 72)
    print("ESTADISTICAS FINALES")
    print("=" * 72)
    print(resumen_global(monitor, meta).to_string(index=False))
    print("\n--- Por ROI ---")
    print(resumen_por_roi(monitor).to_string(index=False))
    alertas = tabla_alertas(monitor)
    if not alertas.empty:
        print(f"\n--- Alertas ({len(alertas)}) ---")
        print(alertas.to_string(index=False))
    print("=" * 72 + "\n")
