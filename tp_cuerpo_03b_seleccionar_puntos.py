"""
tp_cuerpo_03b_seleccionar_puntos.py  --  CRISP-DM fase 3: SELECCION DE PUNTOS DE POSE Y MANOS (canal CUERPO)

Se usan todos los puntos que aportan informacion de movimiento corporal y se descartan, uno por uno y con motivo escrito, los
que no sirven o dañan el modelo. No se elige a mano: se aplican reglas generales SIN usar las etiquetas de mimetismo (no hay
filtracion de informacion de prueba; Hastie, Tibshirani y Friedman, 2009, sec. 7.10.2; Guyon y Elisseeff, 2003). La tabla
tp_cuerpo_03b_puntos.csv deja la decision y el motivo de CADA punto para citarla en el informe.

REGLAS
  B-S1  POSE, visibilidad: se excluye el punto de pose cuya fraccion de cuadros visibles (visibilidad >= 0,5, convencion de
        MediaPipe) promediada entre sesiones es < 0,70. Motivo: un punto casi siempre oculto no mide nada; MediaPipe lo
        extrapola. Es el mismo 70 % que se exige para que un segmento sea usable (D-05). Afecta a caderas y piernas: la
        persona esta sentada tras una mesa y el dataset declara que el estudio se centra en los 25 puntos del tronco
        superior (diccionario de datos, Miranda 2025).
  B-S2  POSE, cabeza: se excluyen los ojos (1 a 6) y las comisuras de la boca (9 y 10). Motivo: la cara ya se modela con su
        propia malla de 468 puntos (canal cara); en la pose estos puntos estan a ~10 a 20 px entre si (con ~200 px de ancho
        de hombros), se mueven con la cabeza como un solo bloque y repiten la informacion de la nariz (se reporta su correlacion
        con la nariz). Se conservan la nariz (0) y las orejas (7, 8) como posicion e inclinacion de la cabeza.
  B-S3  MANOS, palma: se excluyen las bases de los dedos (5, 9, 13, 17), que forman la palma rigida. Motivo: relativas a la
        muneca cambian solo cuando gira la mano; esa orientacion ya queda codificada en las posiciones de los demas puntos.
        Esos 4 puntos son ademas el GRUPO DE CONTROL de la regla siguiente (analogo a los puntos rigidos de la cara).
  B-S4  MANOS, ruido: se excluye el punto de mano cuyo movimiento (RMS relativo a la muneca, solo en cuadros con la mano
        detectada) dividido por el RMS mediano del control, con mediana entre sesiones, es <= 1. Motivo: un punto que se mueve
        como la palma no aporta forma de mano, solo ruido (misma logica que la regla S3 de la cara).
  B-S5  POSE, dedos: se excluyen menique, indice y pulgar de la pose (17 a 22). Motivo: el modelo de manos entrega los 21 puntos
        de cada mano con mas detalle; en la pose esos 6 puntos se ven entre 65 % y 75 % de los cuadros, cerca del corte de B-S1,
        y un corte por visibilidad dejaria el lado izquierdo fuera y el derecho dentro (asimetria sin sentido fisico).
  Se conservan siempre: hombros, codos y munecas de la pose (11 a 16), posicion de cada muneca de mano en el marco del
  cuerpo, presencia de cada mano y los 2 canales de torso.
Lo que NO se hace (y por que): no se elige por correlacion con la etiqueta ni por rendimiento del modelo (seria filtrar
informacion de prueba); no se usa z (inestable); la comparacion 'todos' vs 'seleccion' del script 04 mide si hizo falta.

Uso (despues del script 02, antes del 04):
    python tp_cuerpo_03b_seleccionar_puntos.py
Salidas (salidas_tp_cuerpo/): tp_cuerpo_03b_puntos.csv, tp_cuerpo_03b_puntos.json, tp_cuerpo_03b_seleccionar_puntos.txt
"""
import os, json, glob, warnings
import numpy as np, pandas as pd
import tp_util as u

warnings.filterwarnings("ignore")
ap = u.parser("Seleccion de puntos de pose y manos", root=False, anot=False, salida=u.SALIDA_CUERPO)
ap.add_argument("--vis_min", type=float, default=0.70, help="B-S1: visibilidad media minima de un punto de pose")
a = ap.parse_args()
log = u.Log(); p = log.p
NP, NM = u.N_POSE, u.N_MANO
OFF_I = 2 * NP                              # primer canal de la mano izquierda
OFF_D = OFF_I + 2 * len(u.PTS_MANO)         # primer canal de la mano derecha
OFF_EXTRA = OFF_D + 2 * len(u.PTS_MANO)     # posicion de munecas (4), presencia (2), torso (2)
assert OFF_EXTRA + 8 == u.N_CANALES_CUERPO_TODOS
CABEZA_REDUNDANTE = [1, 2, 3, 4, 5, 6, 9, 10]
DEDOS_POSE = [17, 18, 19, 20, 21, 22]      # menique, indice y pulgar de la pose: los cubre el modelo de manos (21 puntos por mano)
PALMA = [5, 9, 13, 17]
canal_pose = lambda k: [2 * k, 2 * k + 1]
canal_mano = lambda off, k: [off + 2 * (k - 1), off + 2 * (k - 1) + 1]

archivos = sorted(glob.glob(os.path.join(a.salida, "tp_cuerpo_02_[0-9][0-9][AB].npz")))
if not archivos:
    raise SystemExit("No hay archivos tp_cuerpo_02_*.npz: corre antes tp_cuerpo_02_preparar_series.py")
VIS, RAZ, CORR = [], [], []
for f in archivos:
    z = np.load(f); X = z["X_todos"]; tr = z["track"]
    VIS.append(z["vis_pose"].mean(0))
    for t in np.unique(tr):
        x = X[tr == t].astype(np.float32)                                  # segmentos x 29 x canales
        x = x.reshape(-1, x.shape[2])
        # correlacion de cada punto de pose con la nariz (x con x, y con y)
        c = [0.5 * (np.corrcoef(x[:, 2 * k], x[:, 0])[0, 1] + np.corrcoef(x[:, 2 * k + 1], x[:, 1])[0, 1]) if x[:, 2 * k].std() > 0 else np.nan for k in range(NP)]
        CORR.append(c)
    # RMS de cada punto de mano, solo cuadros con la mano detectada
    r = np.full((2, NM - 1), np.nan)
    for hi, (off, col_pres) in enumerate(((OFF_I, OFF_EXTRA + 4), (OFF_D, OFF_EXTRA + 5))):
        pres = X[:, :, col_pres].reshape(-1) > 0.5
        if pres.sum() < 100:
            continue
        M = X[:, :, off:off + 2 * len(u.PTS_MANO)].astype(np.float32).reshape(-1, len(u.PTS_MANO), 2)[pres]
        r[hi] = np.sqrt((M ** 2).sum(2).mean(0))
    ctrl = np.nanmedian(r[:, [k - 1 for k in PALMA]], axis=1, keepdims=True)
    RAZ.append(np.nanmedian(r / ctrl, axis=0))                             # mediana entre las dos manos
VIS = np.mean(VIS, axis=0); CORR = np.nanmedian(np.array(CORR, dtype=float), axis=0)
RAZ = np.array(RAZ); razon_mano = np.nanmedian(RAZ, axis=0)
p(f"Sesiones usadas: {len(archivos)} ({' '.join(os.path.basename(f)[13:16] for f in archivos)})")

filas = []
for k in range(NP):
    if k in DEDOS_POSE:
        d, m = "EXCLUIDO", f"B-S5: dedo de la pose (visible {VIS[k] * 100:.0f} %); ya cubierto por el modelo de manos, que es mas preciso"
    elif VIS[k] < a.vis_min:
        d, m = "EXCLUIDO", f"B-S1: visible solo el {VIS[k] * 100:.0f} % de los cuadros (< {a.vis_min * 100:.0f} %)"
    elif k in CABEZA_REDUNDANTE:
        d, m = "EXCLUIDO", f"B-S2: punto de cabeza ya cubierto por la malla de cara; correlacion con la nariz {CORR[k]:.2f}"
    else:
        d, m = "CONSERVADO", "nariz/orejas: posicion e inclinacion de cabeza" if k in (0, 7, 8) else f"visible el {VIS[k] * 100:.0f} % de los cuadros"
    filas.append(dict(tipo="pose", punto=k, region="cabeza" if k <= 10 else ("tronco_brazos" if k <= 22 else ("caderas" if k <= 24 else "piernas")),
                      indicador=round(float(VIS[k]), 3), decision=d, motivo=m))
for k in u.PTS_MANO:
    r_ = razon_mano[k - 1]
    if k in PALMA:
        d, m = "EXCLUIDO", "B-S3: base de dedo (palma rigida); su orientacion ya esta en los demas puntos; es el control de B-S4"
    elif not np.isfinite(r_) or r_ <= 1.0:
        d, m = "EXCLUIDO", f"B-S4: se mueve como la palma (razon {r_:.2f} <= 1)"
    else:
        d, m = "CONSERVADO", f"se mueve {r_:.2f} veces el control"
    filas.append(dict(tipo="mano", punto=k, region="pulgar" if k <= 4 else ["indice", "medio", "anular", "menique"][(k - 5) // 4], indicador=round(float(r_), 3), decision=d, motivo=m))
T = pd.DataFrame(filas)
T.to_csv(os.path.join(a.salida, "tp_cuerpo_03b_puntos.csv"), index=False)

pose_ok = T[(T.tipo == "pose") & (T.decision == "CONSERVADO")]["punto"].tolist()
mano_ok = T[(T.tipo == "mano") & (T.decision == "CONSERVADO")]["punto"].tolist()
cols = [c for k in pose_ok for c in canal_pose(k)] + [c for off in (OFF_I, OFF_D) for k in mano_ok for c in canal_mano(off, k)] + list(range(OFF_EXTRA, OFF_EXTRA + 8))
json.dump(dict(pose=pose_ok, mano=mano_ok, canales=cols, n_canales=len(cols)), open(os.path.join(a.salida, "tp_cuerpo_03b_puntos.json"), "w"))

p("\n" + "=" * 78 + "\nPOSE: decision por punto\n" + "=" * 78)
p(T[T.tipo == "pose"][["punto", "region", "indicador", "decision", "motivo"]].to_string(index=False))
p("\n" + "=" * 78 + "\nMANO: decision por punto (se aplica a las dos manos)\n" + "=" * 78)
p(T[T.tipo == "mano"][["punto", "region", "indicador", "decision", "motivo"]].to_string(index=False))
p(f"\nPose: {len(pose_ok)} de {NP} puntos | Mano: {len(mano_ok)} de {len(u.PTS_MANO)} puntos por mano (la muneca es el origen)")
p(f"Canales por persona: {len(cols)} ({2 * len(pose_ok)} de pose + {4 * len(mano_ok)} de manos + 8 de muneca/presencia/torso) de {u.N_CANALES_CUERPO_TODOS}")
borde = T[(T.tipo == "mano") & (T.decision == "CONSERVADO") & (T.indicador < 1.3)]
if len(borde):
    p(f"Puntos de mano conservados con margen bajo (razon < 1.3): {borde['punto'].tolist()}")
log.guardar(os.path.join(a.salida, "tp_cuerpo_03b_seleccionar_puntos.txt"))
