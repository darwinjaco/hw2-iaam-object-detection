"""
Pruebas de la logica de permanencia con detecciones sinteticas.

No requieren ultralytics ni GPU: `ROIMonitor` es independiente del detector,
que es justamente el motivo de haberlo separado. Ejecutar:

    python src/test_roi_logic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ROI
from roi_tracker import Detection, ROIMonitor
from stats import resumen_global, resumen_por_roi, tabla_permanencia

W, H, FPS = 1000, 1000, 10.0

ROI_A = ROI(
    id="zona_a",
    name="Zona A",
    points_norm=((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)),
    alert_seconds=2.0,
)
ROI_B = ROI(
    id="zona_b",
    name="Zona B",
    points_norm=((0.0, 0.85), (1.0, 0.85), (1.0, 1.0), (0.0, 1.0)),
)


def caja(cx, cy, lado=40):
    return (cx - lado / 2, cy - lado / 2, cx + lado / 2, cy + lado / 2)


def det(tid, cx, cy, cls=2, conf=0.9):
    return Detection(track_id=tid, cls_id=cls, conf=conf, xyxy=caja(cx, cy))


def nuevo_monitor(enter=3, salir=5, rois=(ROI_A, ROI_B)):
    return ROIMonitor(list(rois), frame_size=(W, H), fps=FPS,
                      enter_frames=enter, exit_frames=salir)


def check(nombre, cond, detalle=""):
    print(f"  [{'OK  ' if cond else 'FALLA'}] {nombre}" + (f" -> {detalle}" if detalle and not cond else ""))
    return bool(cond)


def test_entrada_simple():
    print("\n1. Entrada, permanencia y salida de una ROI")
    m = nuevo_monitor()
    f = 0
    for _ in range(20):                       # dentro de Zona A
        m.update(f, [det(1, 500, 500)]); f += 1
    for _ in range(20):                       # fuera
        m.update(f, [det(1, 950, 100)]); f += 1
    m.finalize()

    st = m.records[1].roi_states["zona_a"]
    ok = check("una sola visita", st.visits == 1, f"visitas={st.visits}")
    # 20 frames dentro: la compensacion de histeresis debe devolver 20 exactos,
    # ni inflados por la espera de salida ni recortados por la de entrada.
    ok &= check("permanencia exacta (sin sesgo de histeresis)",
                st.frames_inside == 20, f"frames={st.frames_inside}, esperado 20")
    ok &= check("visita mas larga = 20 frames", st.max_visit_frames == 20,
                f"max={st.max_visit_frames}")
    ok &= check("salida confirmada", st.inside is False)
    ok &= check("no entro a Zona B", m.records[1].roi_states["zona_b"].visits == 0)
    return ok


def test_histeresis():
    print("\n2. Histeresis: el parpadeo en el borde no genera visitas falsas")
    m_sin = nuevo_monitor(enter=1, salir=1)
    m_con = nuevo_monitor(enter=3, salir=5)
    f = 0
    for i in range(40):                       # alterna dentro/fuera cada frame
        cx = 500 if i % 2 == 0 else 150       # 150 esta fuera del poligono
        m_sin.update(f, [det(1, cx, 500)])
        m_con.update(f, [det(1, cx, 500)])
        f += 1
    m_sin.finalize(); m_con.finalize()

    v_sin = m_sin.records[1].roi_states["zona_a"].visits
    v_con = m_con.records[1].roi_states["zona_a"].visits
    return check("la histeresis suprime las visitas espurias",
                 v_con < v_sin and v_con <= 1, f"sin={v_sin} con={v_con}")


def test_reingreso():
    print("\n3. Reingreso: dos visitas separadas del mismo objeto")
    m = nuevo_monitor()
    f = 0
    for _ in range(15): m.update(f, [det(1, 500, 500)]); f += 1   # visita 1
    for _ in range(15): m.update(f, [det(1, 950, 100)]); f += 1   # sale
    for _ in range(15): m.update(f, [det(1, 500, 500)]); f += 1   # visita 2
    m.finalize()
    st = m.records[1].roi_states["zona_a"]
    return check("dos visitas registradas", st.visits == 2, f"visitas={st.visits}")


def test_alerta():
    print("\n4. Alerta por permanencia excesiva (umbral 2.0 s a 10 FPS)")
    m = nuevo_monitor()
    f = 0
    disparos = 0
    for _ in range(40):
        snap = m.update(f, [det(1, 500, 500)]); f += 1
        disparos += len(snap.new_alerts)
    m.finalize()
    ok = check("se emitio exactamente una alerta por visita", disparos == 1, f"alertas={disparos}")
    ok &= check("la alerta se registro en el historial", len(m.alerts) == 1)
    if m.alerts:
        a = m.alerts[0]
        ok &= check("la alerta dispara cerca del umbral",
                    abs(a.dwell_s - 2.0) < 0.25, f"dwell={a.dwell_s:.2f}s")
    return ok


def test_multiples_rois_y_ocupacion():
    print("\n5. Multiples ROIs simultaneas y ocupacion maxima")
    m = nuevo_monitor()
    f = 0
    for _ in range(15):
        m.update(f, [det(1, 400, 500), det(2, 600, 500), det(3, 500, 950)]); f += 1
    m.finalize()
    ok = check("2 objetos simultaneos en Zona A", m.max_occupancy["zona_a"] == 2,
               f"{m.max_occupancy['zona_a']}")
    ok &= check("1 objeto en Zona B", m.max_occupancy["zona_b"] == 1)
    ok &= check("3 objetos rastreados en total", len(m.records) == 3)
    return ok


def test_objeto_desaparece():
    print("\n6. El objeto abandona el encuadre: la visita se cierra")
    m = nuevo_monitor()
    f = 0
    for _ in range(15): m.update(f, [det(1, 500, 500)]); f += 1
    for _ in range(15): m.update(f, [])          # ninguna deteccion
    f += 15
    m.finalize()
    st = m.records[1].roi_states["zona_a"]
    return check("no sigue contabilizado como dentro", st.inside is False)


def test_tablas():
    print("\n7. Consistencia de las tablas de estadisticas")
    m = nuevo_monitor()
    f = 0
    for _ in range(30):
        m.update(f, [det(1, 500, 500), det(2, 500, 950, cls=7)]); f += 1
    m.finalize()

    perm = tabla_permanencia(m)
    resumen = resumen_por_roi(m)
    glob = resumen_global(m).set_index("metrica")["valor"]

    ok = check("la tabla de permanencia tiene 2 filas", len(perm) == 2, f"{len(perm)}")
    ok &= check("el resumen tiene una fila por ROI", len(resumen) == 2)
    ok &= check("total de objetos = 2", int(glob["total_objetos_detectados"]) == 2)
    ok &= check("ambos entraron a alguna ROI",
                int(glob["objetos_que_entraron_a_alguna_roi"]) == 2)
    ok &= check("la clase truck aparece en el resumen",
                "objetos_clase_truck" in glob.index)
    ok &= check("los tiempos son positivos", (perm["tiempo_en_roi_s"] > 0).all())
    return ok


def test_sin_detecciones():
    print("\n8. Caso borde: video sin ninguna deteccion")
    m = nuevo_monitor()
    for f in range(10):
        m.update(f, [])
    m.finalize()
    perm = tabla_permanencia(m)
    resumen = resumen_por_roi(m)
    glob = resumen_global(m).set_index("metrica")["valor"]
    ok = check("tabla de permanencia vacia sin excepcion", perm.empty)
    ok &= check("resumen por ROI sigue teniendo 2 filas", len(resumen) == 2)
    ok &= check("total de objetos = 0", int(glob["total_objetos_detectados"]) == 0)
    return ok


def main() -> int:
    print("=" * 66)
    print("PRUEBAS DE LA LOGICA DE PERMANENCIA EN ROI")
    print("=" * 66)
    pruebas = [
        test_entrada_simple, test_histeresis, test_reingreso, test_alerta,
        test_multiples_rois_y_ocupacion, test_objeto_desaparece,
        test_tablas, test_sin_detecciones,
    ]
    resultados = [t() for t in pruebas]
    print("\n" + "=" * 66)
    print(f"RESULTADO: {sum(resultados)}/{len(resultados)} pruebas superadas")
    print("=" * 66)
    return 0 if all(resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())
