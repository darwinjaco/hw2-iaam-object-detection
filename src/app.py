"""
Interfaz interactiva (Streamlit) del sistema de monitoreo por ROI.

Ejecutar desde la raiz del proyecto:
    streamlit run src/app.py

La app no reimplementa el pipeline: consume el mismo generador
`roi_tracker.process_video` que usa la CLI, de modo que los resultados
mostrados aqui y los del script son identicos por construccion.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import stats as st_mod
from roi_editor import dibujar_rois, leer_frame
from roi_tracker import process_video

st.set_page_config(page_title="Monitoreo por ROI | YOLO + ByteTrack", layout="wide")

# --------------------------------------------------------------------------
# Estado de sesion
# --------------------------------------------------------------------------
for clave, valor in {
    "monitor": None,
    "meta": {},
    "video_salida": None,
    "procesado": False,
}.items():
    st.session_state.setdefault(clave, valor)


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
st.sidebar.title("Configuracion")

st.sidebar.subheader("Video")
subido = st.sidebar.file_uploader("Subir video", type=["mp4", "avi", "mov", "mkv"])
if subido is not None:
    tmp = Path(tempfile.gettempdir()) / subido.name
    tmp.write_bytes(subido.getbuffer())
    ruta_video = tmp
else:
    ruta_video = cfg.VIDEO_IN_DEFAULT
    st.sidebar.caption(f"Usando por defecto: `{ruta_video.name}`")

st.sidebar.subheader("Modelo")
modelo = st.sidebar.text_input("Pesos YOLO", value=cfg.MODEL_NAME)
clases_sel = st.sidebar.multiselect(
    "Clases COCO",
    options=list(cfg.COCO_NAMES.keys()),
    default=cfg.TARGET_CLASSES,
    format_func=lambda c: f"{c} - {cfg.COCO_NAMES[c]}",
)
conf = st.sidebar.slider("Umbral de confianza", 0.05, 0.95, cfg.CONF_THRES, 0.05)
imgsz = st.sidebar.select_slider("Tamano de inferencia", options=[640, 960, 1280, 1600, 2560, 3200, 4096], value=cfg.IMGSZ)
st.sidebar.subheader("Logica de ROI")
anchor = st.sidebar.radio(
    "Punto de referencia del box",
    options=["centroid", "bottom"],
    index=0 if cfg.ANCHOR_MODE == "centroid" else 1,
    help="centroid: vista cenital. bottom: vista oblicua (aproxima el contacto con el suelo).",
)
enter_frames = st.sidebar.number_input("Frames para confirmar ENTRADA", 1, 30, cfg.ENTER_CONSEC_FRAMES)
exit_frames = st.sidebar.number_input("Frames para confirmar SALIDA", 1, 60, cfg.EXIT_CONSEC_FRAMES)

st.sidebar.subheader("Ejecucion")
max_frames = st.sidebar.number_input(
    "Limite de frames (0 = todo el video)", 0, 100000, 0, step=30,
    help="Util para iterar rapido mientras ajustas las ROIs.",
)
paso_vista = st.sidebar.slider("Refrescar vista cada N frames", 1, 30, 5)
guardar_video = st.sidebar.checkbox("Guardar video anotado", value=True)


# --------------------------------------------------------------------------
# Carga de ROIs
# --------------------------------------------------------------------------
try:
    rois_actuales = cfg.load_rois()
except Exception as exc:  # noqa: BLE001
    st.sidebar.error(f"Error leyendo rois.json: {exc}")
    rois_actuales = list(cfg.DEFAULT_ROIS)


st.title("Monitoreo de objetos en Regiones de Interes")
st.caption("YOLO + ByteTrack | conteo, tiempo de permanencia, alertas y estadisticas por ROI")

tab_roi, tab_run, tab_res = st.tabs(
    ["1. Definir y verificar ROIs", "2. Procesar video", "3. Resultados y exportacion"]
)

# --------------------------------------------------------------------------
# Tab 1: ROIs
# --------------------------------------------------------------------------
with tab_roi:
    col_cfg, col_prev = st.columns([1, 1.6])

    with col_cfg:
        st.subheader("Definicion de ROIs")
        st.markdown(
            "Coordenadas **normalizadas** `[0,1]`: la configuracion es independiente "
            "de la resolucion. `alert_seconds: null` desactiva la alerta de permanencia."
        )
        with st.expander("Editar rois.json", expanded=False):
            texto_json = st.text_area(
                "rois.json",
                value=json.dumps({"rois": [r.to_dict() for r in rois_actuales]}, indent=2, ensure_ascii=False),
                height=380,
            )
        c1, c2 = st.columns(2)
        if c1.button("Aplicar y guardar", width="stretch"):
            try:
                data = json.loads(texto_json)
                nuevas = [cfg.ROI.from_dict(d) for d in data["rois"]]
                cfg.save_rois(nuevas)
                st.success(f"{len(nuevas)} ROIs guardadas.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"JSON invalido: {exc}")
        if c2.button("Restaurar defaults", width="stretch"):
            cfg.save_rois(cfg.DEFAULT_ROIS)
            st.rerun()

        st.info(
            "Las ROIs por defecto son una **propuesta inicial**. Verificalas en la "
            "vista previa antes de reportar resultados en el informe."
        )

    with col_prev:
        st.subheader("Verificacion sobre un frame real")
        try:
            cap = cv2.VideoCapture(str(ruta_video))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            cap.release()
            idx = st.slider("Frame de referencia", 0, max(0, total - 1), min(200, total - 1))
            rejilla = st.checkbox("Mostrar rejilla normalizada", value=True)
            frame = leer_frame(Path(ruta_video), idx)
            st.image(
                cv2.cvtColor(dibujar_rois(frame, rois_actuales, con_rejilla=rejilla), cv2.COLOR_BGR2RGB),
                width="stretch",
                caption=f"Frame {idx} | {frame.shape[1]}x{frame.shape[0]}",
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo generar la vista previa: {exc}")

# --------------------------------------------------------------------------
# Tab 2: Procesamiento
# --------------------------------------------------------------------------
with tab_run:
    st.subheader("Procesamiento")
    if not clases_sel:
        st.warning("Selecciona al menos una clase COCO en la barra lateral.")

    if st.button("Iniciar procesamiento", type="primary", disabled=not clases_sel):
        salida = cfg.VIDEO_OUT_DEFAULT if guardar_video else None
        barra = st.progress(0.0, text="Cargando modelo...")
        col_video, col_live = st.columns([2, 1])
        marco = col_video.empty()
        panel = col_live.empty()
        log_alertas = st.empty()
        alertas_vistas: list[str] = []

        monitor = None
        meta: dict = {}
        try:
            gen = process_video(
                video_in=Path(ruta_video),
                rois=rois_actuales,
                video_out=salida,
                model_name=modelo,
                classes=clases_sel,
                conf=conf,
                imgsz=int(imgsz),
                anchor_mode=anchor,
                enter_frames=int(enter_frames),
                exit_frames=int(exit_frames),
                max_frames=int(max_frames) or None,
            )
            for frame_anot, snap, monitor, meta in gen:
                total = meta.get("total_frames") or 0
                if int(max_frames):
                    total = min(total or int(max_frames), int(max_frames))

                for a in snap.new_alerts:
                    alertas_vistas.append(
                        f"t={a.timestamp_s:6.2f}s - ID {a.track_id} ({a.cls_name}) "
                        f"lleva {a.dwell_s:.1f}s en '{a.roi_name}'"
                    )

                if snap.frame_idx % paso_vista == 0:
                    prog = (snap.frame_idx + 1) / total if total else 0.0
                    barra.progress(
                        min(1.0, prog),
                        text=f"Frame {snap.frame_idx + 1}/{total or '?'} - {meta['proc_fps']:.1f} FPS",
                    )
                    marco.image(cv2.cvtColor(frame_anot, cv2.COLOR_BGR2RGB), width="stretch")
                    panel.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "ROI": r.name,
                                    "Dentro ahora": len(snap.occupancy.get(r.id, [])),
                                    "Objetos unicos": sum(
                                        1 for rec in monitor.records.values()
                                        if rec.roi_states[r.id].visits > 0
                                    ),
                                }
                                for r in rois_actuales
                            ]
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                    if alertas_vistas:
                        log_alertas.warning(
                            f"Alertas emitidas: {len(alertas_vistas)}\n\n"
                            + "\n\n".join(alertas_vistas[-5:])
                        )
            barra.progress(1.0, text="Procesamiento completado.")
        except FileNotFoundError as exc:
            st.error(f"{exc}")
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

        if monitor is not None and monitor.records:
            st.session_state.monitor = monitor
            st.session_state.meta = meta
            st.session_state.video_salida = salida
            st.session_state.procesado = True
            st.success(
                f"Listo: {meta.get('frame_idx', 0) + 1} frames, "
                f"{len(monitor.records)} objetos rastreados, "
                f"{len(monitor.alerts)} alertas."
            )
        elif monitor is not None:
            st.warning("No se rastreo ningun objeto. Baja el umbral de confianza o revisa las clases.")

    if st.session_state.procesado:
        st.info("Resultados disponibles en la pestana 3.")

# --------------------------------------------------------------------------
# Tab 3: Resultados
# --------------------------------------------------------------------------
with tab_res:
    monitor = st.session_state.monitor
    if monitor is None:
        st.info("Aun no hay resultados. Ejecuta el procesamiento en la pestana 2.")
    else:
        meta = st.session_state.meta
        g = st_mod.resumen_global(monitor, meta).set_index("metrica")["valor"]
        por_roi = st_mod.resumen_por_roi(monitor)
        permanencia = st_mod.tabla_permanencia(monitor)
        alertas = st_mod.tabla_alertas(monitor)

        st.subheader("Estadisticas finales")
        c = st.columns(4)
        c[0].metric("Objetos detectados", int(g.get("total_objetos_detectados", 0)))
        c[1].metric("Entraron a alguna ROI", int(g.get("objetos_que_entraron_a_alguna_roi", 0)))
        c[2].metric("Permanencia promedio", f"{float(g.get('tiempo_promedio_permanencia_global_s', 0)):.2f} s")
        c[3].metric("Alertas", int(g.get("alertas_totales", 0)))

        c2 = st.columns(3)
        c2[0].metric("Visitas totales", int(g.get("visitas_totales_todas_las_rois", 0)))
        c2[1].metric("FPS de procesamiento", f"{float(g.get('fps_procesamiento', 0)):.1f}")
        c2[2].metric("Frames procesados", int(g.get("frames_procesados", 0)))

        st.markdown("### Resumen por ROI")
        st.dataframe(por_roi, hide_index=True, width="stretch")

        if not por_roi.empty:
            col_a, col_b = st.columns(2)
            col_a.markdown("**Visitas por ROI**")
            col_a.bar_chart(por_roi.set_index("roi_nombre")["visitas_totales"])
            col_b.markdown("**Permanencia promedio por ROI (s)**")
            col_b.bar_chart(por_roi.set_index("roi_nombre")["tiempo_promedio_por_objeto_s"])

        st.markdown("### Permanencia por objeto")
        st.dataframe(permanencia, hide_index=True, width="stretch")

        st.markdown("### Alertas")
        if alertas.empty:
            st.caption("Ningun objeto supero el umbral de permanencia configurado.")
        else:
            st.dataframe(alertas, hide_index=True, width="stretch")

        st.markdown("### Exportacion")
        if st.button("Escribir CSV y XLSX en outputs/"):
            escritos = st_mod.exportar(monitor, cfg.OUTPUT_DIR, meta=meta)
            st.success("Archivos generados:\n\n" + "\n\n".join(f"- `{v}`" for v in escritos.values()))

        d = st.columns(4)
        tablas = {
            "resumen_por_roi.csv": por_roi,
            "permanencia_por_objeto.csv": permanencia,
            "objetos.csv": st_mod.tabla_objetos(monitor),
            "alertas.csv": alertas,
        }
        for col, (nombre, df) in zip(d, tablas.items()):
            col.download_button(
                nombre,
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name=nombre,
                mime="text/csv",
                width="stretch",
            )

        salida = st.session_state.video_salida
        if salida and Path(salida).exists():
            st.download_button(
                "Descargar video anotado",
                data=Path(salida).read_bytes(),
                file_name=Path(salida).name,
                mime="video/mp4",
            )
