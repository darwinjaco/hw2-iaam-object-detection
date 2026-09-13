# Informe técnico — Monitoreo de vehículos en Regiones de Interés mediante YOLO y ByteTrack

**Materia:** Inteligencia Artificial Aplicada a la Manufactura (IAAM)
**Tarea:** HW2 — Sistema de visión por computador con ROI
**Autor:** `[COMPLETAR: nombre]`
**Fecha:** `[COMPLETAR]`

> **Instrucción de uso de esta plantilla.** Los campos marcados `[COMPLETAR: ...]`
> requieren ejecutar el sistema o realizar la verificación manual descrita. **No
> deben rellenarse con valores estimados**: un informe de ingeniería que reporta
> números no medidos es un informe inválido, con independencia de lo bien
> redactado que esté. Borre este bloque antes de entregar.

---

## 1. Descripción del problema

Los estacionamientos de superficie de centros comerciales y campus operan sin
instrumentación: la ocupación de las bahías y el estado de los pasillos de
circulación solo se conocen por inspección visual de un operador o por sensores
puntuales (espiras magnéticas, sensores ultrasónicos por plaza), cuyo costo de
instalación y mantenimiento escala linealmente con el número de plazas.

Dos fenómenos concretos degradan la operación y no son observables con la
infraestructura habitual:

1. **Ocupación desconocida en tiempo real.** Sin un conteo continuo de vehículos
   por zona no es posible dirigir a los conductores hacia los sectores con
   disponibilidad, lo que genera circulación innecesaria en busca de plaza.
2. **Obstrucción de pasillos de circulación.** Un vehículo detenido en un pasillo
   —esperando que se libere una plaza, en doble fila o descargando— bloquea el
   flujo y, en un pasillo de sentido único, propaga la congestión hacia la
   entrada. La detención en sí no es anómala; lo es su *duración*.

Una cámara cenital ya instalada por seguridad cubre decenas de plazas
simultáneamente. El problema es, por tanto, de procesamiento, no de sensado:
convertir un flujo de video existente en indicadores de ocupación y permanencia
por zona.

## 2. Justificación del escenario elegido

Se selecciona el escenario **Estacionamientos Inteligentes** por cuatro razones:

1. **Compatibilidad con el modelo preentrenado.** Los vehículos pertenecen a
   clases del vocabulario COCO (`car`, `motorcycle`, `bus`, `truck`) con las que
   YOLO26 fue entrenado. No se requiere reentrenamiento ni etiquetado, lo que
   mantiene el alcance dentro del laboratorio de clase.
2. **La vista cenital elimina el sesgo de proyección.** En vista oblicua, el
   centroide de un bounding box no corresponde a la posición del objeto en el
   plano del suelo, y la pertenencia a una ROI depende de la altura del vehículo.
   En vista nadir esa ambigüedad desaparece: el centroide del box aproxima
   directamente la posición del vehículo sobre el pavimento.
3. **El escenario exige permanencia, no solo cruce.** Es lo que distingue una
   solución de monitoreo de un simple contador de línea: en un estacionamiento la
   pregunta relevante no es cuántos vehículos pasaron, sino cuántos hay y desde
   hace cuánto.
4. **Distintas ROIs con semántica distinta.** El mismo pipeline debe tratar una
   zona donde permanecer es lo normal (bahías) y zonas donde permanecer es una
   anomalía (pasillos). Esto justifica que el umbral de alerta sea un parámetro
   por ROI y no una constante global.

### 2.1 Datos de entrada

| Propiedad | Valor |
|---|---|
| Archivo | `videos/input/Track.mp4` |
| Resolución | 1920 × 1080 |
| Tasa de cuadros | 29.97 fps |
| Duración | 28.83 s (864 frames) |
| Punto de vista | Cenital (dron), cámara estática |
| Escena | Estacionamiento de superficie: pasillos de circulación y bahías |

> `[COMPLETAR: procedencia del video — fuente, licencia y fecha de captura.]`
> Este dato es exigible: un informe no puede omitir el origen de sus datos.

## 3. Definición de la Región de Interés

### 3.1 Representación

Cada ROI es un **polígono convexo o cóncavo de N vértices en coordenadas
normalizadas** `[0, 1]` respecto al ancho y alto del frame, persistido en
`config/rois.json`. La normalización desacopla la configuración de la resolución:
las mismas ROIs son válidas si la cámara se sustituye por una de 4K.

La pertenencia se evalúa con `cv2.pointPolygonTest`, que resuelve correctamente
polígonos no rectangulares —necesario porque los pasillos reales no son
rectángulos alineados con los ejes de la imagen.

### 3.2 Punto de referencia del objeto

Se evalúa el **centroide** del bounding box contra el polígono, justificado por la
vista cenital (sección 2, punto 2). El sistema admite alternativamente el punto
medio del borde inferior (`--anchor bottom`), apropiado para vistas oblicuas donde
ese punto aproxima el contacto del vehículo con el suelo.

### 3.3 ROIs definidas

| ID | Nombre | Función | Umbral de alerta |
|---|---|---|---|
| `pasillo_norte` | Pasillo de circulación norte | Flujo de tránsito; detectar obstrucción | 12 s |
| `bahias_centro` | Bahías de parqueo centrales | Ocupación; permanecer es lo esperado | — (sin alerta) |
| `pasillo_sur` | Pasillo de circulación sur | Flujo de tránsito; detectar obstrucción | 12 s |

> `[COMPLETAR: insertar outputs/roi_preview.jpg y confirmar que cada polígono
> cubre la zona descrita. Generar con:
> python src/roi_editor.py --preview --rejilla]`
>
> `[COMPLETAR: coordenadas finales de cada polígono, copiadas de config/rois.json
> después de la verificación visual.]`

### 3.4 Justificación del umbral de alerta

El umbral de 12 s para los pasillos es un **parámetro de diseño, no una medición**.
Se fija bajo el criterio de que un vehículo en tránsito normal atraviesa el ancho
del pasillo en un tiempo sensiblemente menor, de modo que superar el umbral
implica detención efectiva.

> `[COMPLETAR: sustituir este criterio cualitativo por evidencia. Procedimiento:
> ejecutar el pipeline, tomar la columna `visita_mas_larga_s` de
> outputs/permanencia_por_objeto.csv para los vehículos en tránsito, y fijar el
> umbral por encima del percentil 95 de esa distribución. Reportar el valor
> obtenido y el umbral final adoptado.]`

## 4. Estrategia de conteo y medición de permanencia

### 4.1 Arquitectura

```
frame ──► YOLO26 (detección) ──► ByteTrack (asociación, track_id persistente)
            │
            └─► ROIMonitor: para cada par (track_id, ROI)
                    pointPolygonTest ──► histéresis ──► estado {dentro, visitas,
                                                                frames acumulados}
                    │
                    ├─► Annotator  ──► video anotado / interfaz Streamlit
                    └─► stats      ──► CSV / XLSX
```

`ROIMonitor` mantiene un diccionario `{track_id: {roi_id: ROIState}}`. En cada
frame, para cada objeto rastreado y cada ROI, se evalúa la pertenencia y se
actualiza la máquina de estados.

### 4.2 Definiciones operativas

Estas definiciones son necesarias para que los números reportados sean
interpretables:

- **Objeto único:** un `track_id` distinto asignado por ByteTrack.
- **Visita:** cada transición confirmada `fuera → dentro` de una ROI.
- **Tiempo de permanencia por objeto:** frames confirmados dentro ÷ fps,
  agregando todas sus visitas.
- **Tiempo promedio de permanencia:** media del anterior sobre los objetos con al
  menos una visita a esa ROI. Se reporta además la media *por visita*, que
  difiere de la anterior cuando existen reingresos.
- **Ocupación máxima simultánea:** máximo, sobre todos los frames, del número de
  objetos confirmados dentro de la ROI en ese frame.

### 4.3 Histéresis temporal y corrección de su sesgo

Una prueba de pertenencia aplicada frame a frame sin filtrado es inestable: el
bounding box de un vehículo sobre el borde del polígono oscila entre dentro y
fuera, lo que produce decenas de visitas espurias para un único evento físico y
corrompe tanto el conteo como el tiempo promedio.

Se aplica **histéresis temporal**: una entrada se confirma tras
`ENTER_CONSEC_FRAMES = 3` observaciones consecutivas dentro, y una salida tras
`EXIT_CONSEC_FRAMES = 5` fuera. Los valores son asimétricos deliberadamente:
confirmar la salida es más costoso porque una oclusión breve no debe interpretarse
como abandono de la zona.

La histéresis introduce a su vez un **error sistemático** en el tiempo medido: la
entrada se registra tarde (`ENTER − 1` frames) y la salida también (`EXIT − 1`
frames), con un sesgo neto de `EXIT − ENTER = +2` frames por visita. El sistema lo
compensa explícitamente: al confirmar la entrada acredita retroactivamente los
frames en que el objeto ya estaba dentro, y al confirmar la salida descuenta los
frames contabilizados mientras ya estaba fuera. La prueba 1 de
`src/test_roi_logic.py` verifica que un objeto presente exactamente 20 frames
dentro de la ROI reporta 20 frames, no 22.

### 4.4 Alertas

Cuando la visita en curso de un objeto supera `alert_seconds` de su ROI, se emite
una `Alert` (una sola por visita, no una por frame). La alerta se muestra como
banner en el video, se registra en `outputs/alertas.csv` y aparece en vivo en la
interfaz. El estado de alerta se reinicia al confirmarse la salida.

## 5. Resultados obtenidos

> **Sección pendiente de ejecución.** Todos los valores siguientes se obtienen de
> `outputs/` tras ejecutar `python src/run_pipeline.py`.

### 5.1 Configuración de la ejecución reportada

| Parámetro | Valor |
|---|---|
| Modelo | `yolo26m.pt` (COCO, 80 clases) |
| Tracker | `bytetrack.yaml` (configuración por defecto de Ultralytics) |
| Clases | `[COMPLETAR: 2, 3, 5, 7 o el subconjunto usado]` |
| Umbral de confianza | `[COMPLETAR]` |
| Tamaño de inferencia | `[COMPLETAR]` |
| Histéresis entrada / salida | `[COMPLETAR]` frames |
| Hardware | `[COMPLETAR: CPU/GPU, modelo]` |

### 5.2 Rendimiento

| Métrica | Valor |
|---|---|
| Frames procesados | `[COMPLETAR]` |
| Tiempo total de procesamiento | `[COMPLETAR] s` |
| FPS de procesamiento | `[COMPLETAR]` |
| Razón respecto a tiempo real (FPS_proc / 29.97) | `[COMPLETAR]` |

> Si la razón resulta menor que 1.0, **el informe no puede afirmar operación en
> tiempo real**. Debe reportarse el valor medido y discutirse en las limitaciones
> qué configuración lo alcanzaría (modelo `yolo26n`, `imgsz` menor, exportación a
> TensorRT).

### 5.3 Estadísticas finales

De `outputs/resumen_global.csv`:

| Métrica | Valor |
|---|---|
| Total de objetos detectados (track_id únicos) | `[COMPLETAR]` |
| Objetos que entraron a alguna ROI | `[COMPLETAR]` |
| Visitas totales | `[COMPLETAR]` |
| Tiempo promedio de permanencia global | `[COMPLETAR] s` |
| Alertas emitidas | `[COMPLETAR]` |

De `outputs/resumen_por_roi.csv`:

| ROI | Objetos únicos | Visitas | Permanencia media (s) | Mediana (s) | Máximo (s) | Ocupación máx. | Alertas |
|---|---|---|---|---|---|---|---|
| `pasillo_norte` | | | | | | | |
| `bahias_centro` | | | | | | | |
| `pasillo_sur` | | | | | | | |

### 5.4 Validación contra conteo manual

> **Esta subsección es la que convierte el trabajo en un informe de ingeniería.**
> Sin ella, los números de 5.3 son salidas de un programa sin evidencia de que
> sean correctas.

Procedimiento:

1. Reproducir `videos/input/Track.mp4` con control cuadro a cuadro.
2. Para cada ROI, contar manualmente los vehículos que ingresan durante los 28.83 s.
3. Comparar con la columna `objetos_unicos` de `resumen_por_roi.csv`.
4. Clasificar las discrepancias en: no detectado (falso negativo), detectado sin
   ingresar (falso positivo) y vehículo fragmentado en dos `track_id`
   (conmutación de identidad).

| ROI | Conteo manual | Conteo del sistema | Error absoluto | Error relativo |
|---|---|---|---|---|
| `pasillo_norte` | `[COMPLETAR]` | `[COMPLETAR]` | | |
| `bahias_centro` | `[COMPLETAR]` | `[COMPLETAR]` | | |
| `pasillo_sur` | `[COMPLETAR]` | `[COMPLETAR]` | | |

Análisis cualitativo de los errores: `[COMPLETAR]`

### 5.5 Evidencia visual

> `[COMPLETAR: 2–3 capturas de videos/output/roi_counting.mp4 que muestren
> (a) bounding boxes con ID y tiempo de permanencia, (b) las tres ROIs con su
> contador, (c) una alerta activa.]`

## 6. Limitaciones y posibles mejoras

### 6.1 Limitaciones

**Del método de conteo**

- Un `track_id` no equivale a un vehículo físico. Una oclusión prolongada
  fragmenta un vehículo en dos identificadores, de modo que el conteo de objetos
  únicos es una **cota superior** del conteo real. La magnitud de este efecto solo
  se conoce mediante la validación de 5.4.
- Los vehículos ya estacionados al inicio del video se contabilizan como presentes
  en `bahias_centro` desde el frame 0, sin que exista un evento de entrada
  observado. La ocupación medida en esa ROI mide *estado*, no *flujo*.
- Un vehículo que atraviesa una ROI en menos de `ENTER_CONSEC_FRAMES` frames no se
  registra. A 29.97 fps esto equivale a ~0.1 s; es despreciable a velocidades de
  estacionamiento pero no lo sería en una vía rápida.

**Del detector**

- El modelo es COCO preentrenado, sin ajuste al dominio: no se dispone de métricas
  de precisión y exhaustividad *sobre este video*. Las cifras de mAP publicadas
  para YOLO26 se refieren a COCO y no son transferibles a esta escena.
- En vista cenital, los vehículos oscuros sobre pavimento oscuro y los vehículos
  parcialmente ocultos por vegetación o mobiliario urbano son los casos de fallo
  esperables. `[COMPLETAR: confirmar u refutar tras revisar el video procesado.]`
- Las clases COCO no distinguen vehículos de reparto de automóviles particulares,
  distinción que sí sería relevante en una aplicación real de gestión de
  estacionamiento.

**Del alcance**

- Una sola secuencia, una sola cámara, condiciones diurnas y despejadas. No hay
  evidencia sobre el comportamiento del sistema con lluvia, de noche o con la
  cámara en movimiento.
- Los umbrales de alerta son parámetros fijados por criterio, no calibrados con
  datos (ver 3.4).

### 6.2 Mejoras propuestas

| Mejora | Problema que resuelve |
|---|---|
| Homografía a coordenadas del plano del suelo | Permitiría definir las ROIs en metros y medir velocidad real, no en píxeles. |
| Reidentificación por apariencia (BoT-SORT con ReID) | Reduciría la fragmentación de trayectorias por oclusión, principal fuente de sobreconteo. |
| Ajuste fino con un conjunto etiquetado de la propia cámara | Habilitaría métricas de detección sobre el dominio real, hoy ausentes. |
| Umbrales de alerta calibrados sobre la distribución empírica de permanencias | Sustituiría el criterio cualitativo de 3.4 por evidencia. |
| Exportación a TensorRT / ONNX o uso de `yolo26n` | Necesario si la validación de 5.2 muestra que no se alcanza tiempo real. |
| Persistencia en base de datos temporal en lugar de CSV | Requisito para operación continua; los CSV solo sirven para análisis por lotes. |
| Detección automática de plazas libres por segmentación de la marcación vial | Convertiría el conteo de ocupación en disponibilidad por plaza. |

## 7. Conclusiones

`[COMPLETAR: redactar únicamente después de tener los resultados de la sección 5.
Debe responder si el sistema cumple el objetivo planteado en la sección 1, con qué
exactitud medida y bajo qué condiciones. No incluir afirmaciones que no se
sostengan en los números de 5.3 y 5.4.]`

## 8. Referencias

> `[COMPLETAR: verificar cada referencia antes de incluirla. No citar fuentes no
> consultadas.]`

- Documentación de Ultralytics YOLO26 — `https://docs.ultralytics.com/models/yolo26`
- Documentación de seguimiento de Ultralytics (ByteTrack) — `https://docs.ultralytics.com/modes/track`
- `[COMPLETAR: referencia del artículo original de ByteTrack, si se cita en el texto.]`
- `[COMPLETAR: fuente y licencia del video utilizado.]`
