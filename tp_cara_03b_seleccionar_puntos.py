"""
tp_cara_03b_seleccionar_puntos.py  --  CRISP-DM fase 3 (preparacion de datos): SELECCION DE PUNTOS DE LA MALLA FACIAL

Pregunta del profesor guia: "¿por que no se usan todos los puntos?". Respuesta de diseno: se usan todos los que aportan
informacion de expresion y se descartan, uno por uno y con motivo escrito, los que no sirven o que dañan el modelo.
Este script NO decide a mano punto por punto: aplica 3 reglas generales y deja una tabla con la decision y el motivo de
CADA uno de los 468 puntos (tp_cara_03b_puntos.csv) para citarla en el informe.

REGLAS (todas SIN usar las etiquetas de mimetismo -> no hay filtracion de informacion de prueba al entrenamiento;
        Hastie, Tibshirani y Friedman, 2009, seccion 7.10.2; Guyon y Elisseeff, 2003):
  S1  Se excluyen los 13 puntos RIGIDOS usados para alinear la cara (D-08). Motivo: la alineacion los deja casi quietos por
      construccion, de modo que lo que queda en ellos es error de ajuste, no expresion. Su movimiento real ya esta en los 4
      canales de cabeza (D-11).
  S2  Se excluye el CONTORNO de la cara (36 puntos del ovalo de MediaPipe), salvo 5 puntos de la mandibula (148, 176, 152,
      400, 377). Motivo: el contorno no pertenece a ningun musculo de expresion; su posicion en la imagen depende del giro de la
      cabeza fuera del plano, del pelo, de la barba y de oclusiones (Kartynnik et al., 2019: la malla se ajusta mejor en el
      interior de la cara). La mandibula se conserva porque abrir la boca baja el menton (unidad de accion AU26/27 del FACS;
      Ekman y Friesen, 1978). En el dato medido, el contorno se mueve ~3 veces mas que los labios (sesion 01A), lo que no es
      expresion (es giro de cabeza y ruido).
  S3  Se excluyen los puntos cuyo movimiento no se distingue del de un punto que NO se mueve. Grupo de control = los 9 puntos
      rigidos de la linea media (frente y puente de la nariz: 10, 151, 168, 6, 197, 195, 5, 4, 1). Para cada sesion se mide la
      raiz del promedio de cuadrados (RMS) del desplazamiento de cada punto y se divide por la mediana del RMS del control en
      esa misma sesion. Un punto se descarta si su razon mediana entre sesiones es <= 1. Motivo: con ~66 px entre ojos y
      coordenadas enteras (script 01), un punto que se mueve como el control solo aporta ruido al modelo y diluye a los demas
      (la maldicion de la dimension; Guyon y Elisseeff, 2003).
Lo que NO se hace (y por que): no se elige por correlacion con la etiqueta ni por rendimiento del modelo (seria filtrar
informacion de prueba); no se agrupan puntos vecinos redundantes (podria perder detalle; la comparacion 'todos' vs
'seleccion' del script 04 mide si hacia falta); no se usa z (D-02).

Uso (despues de tp_cara_02, antes de tp_cara_04):
    python tp_cara_03b_seleccionar_puntos.py
Salidas (salidas_tp/): tp_cara_03b_puntos.csv, tp_cara_03b_puntos.json, tp_cara_03b_seleccionar_puntos.txt
"""
import os, json, glob, warnings
import numpy as np, pandas as pd
import tp_util as u

warnings.filterwarnings("ignore")
ap = u.parser("Seleccion de puntos de la malla facial", root=False, anot=False)
a = ap.parse_args()
log = u.Log(); p = log.p

# regiones: conjuntos de puntos de la malla de MediaPipe (FaceLandmarksConnections de MediaPipe, verificados con mediapipe 0.10.32)
REGION = {
    "labios": [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310,
               311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415],
    "ojo_izq": [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466],
    "ojo_der": [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246],
    "ceja_izq": [276, 282, 283, 285, 293, 295, 296, 300, 334, 336],
    "ceja_der": [46, 52, 53, 55, 63, 65, 66, 70, 105, 107],
    "contorno": [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234, 251, 284, 288, 297, 323, 332,
                 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454],
    "nariz": [1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 278, 294, 326, 327, 344, 440],
}
MANDIBULA = [148, 176, 152, 400, 377]
CONTROL = [10, 151, 168, 6, 197, 195, 5, 4, 1]
assert set(CONTROL) <= set(u.RIGIDOS)
region = np.array(["mejilla_frente_otros"] * u.N_LM, dtype=object)
for r_, v in REGION.items():
    region[v] = r_
region[u.RIGIDOS] = "rigido(alineacion)"

# ------------------------------------------------------------------------------------------------ RMS de cada punto por sesion
archivos = sorted(glob.glob(os.path.join(a.salida, "tp_cara_02_[0-9][0-9][AB].npz")))
if not archivos:
    raise SystemExit("No hay archivos tp_cara_02_*.npz: corre antes tp_cara_02_preparar_series.py")
RMS = []; nombres = []
for f in archivos:
    z = np.load(f); X = z["X_todos"][:, :, :2 * u.N_LM]; tr = z["track"]
    r = []
    for t in np.unique(tr):
        x = X[tr == t].astype(np.float32)
        r.append(np.sqrt((x ** 2).mean(axis=(0, 1)).reshape(u.N_LM, 2).sum(1)))      # RMS por punto (x e y juntos)
    RMS.append(np.mean(r, axis=0)); nombres.append(os.path.basename(f)[11:14])
RMS = np.array(RMS)                                                                # sesiones x 468
piso = np.median(RMS[:, CONTROL], axis=1, keepdims=True)                           # control por sesion
RAZON = RMS / piso
razon_med = np.median(RAZON, axis=0)
frac_sobre = (RAZON > 1).mean(axis=0)
p(f"Sesiones usadas: {len(archivos)} ({' '.join(nombres)})")
p(f"RMS del grupo de control (9 puntos rigidos de la linea media), mediana entre sesiones: {np.median(piso):.3f} (unidades normalizadas)")

# ------------------------------------------------------------------------------------------------ decisiones
dec, motivo = [], []
for k in range(u.N_LM):
    if k in u.RIGIDOS:
        dec.append("EXCLUIDO"); motivo.append("S1: punto rigido usado para alinear la cara (casi quieto por construccion)")
    elif region[k] == "contorno" and k not in MANDIBULA:
        dec.append("EXCLUIDO"); motivo.append("S2: contorno de la cara (depende del giro de cabeza, pelo y oclusion, no de la expresion)")
    elif razon_med[k] <= 1.0:
        dec.append("EXCLUIDO"); motivo.append(f"S3: se mueve como un punto sin movimiento (razon {razon_med[k]:.2f} <= 1 respecto del control)")
    else:
        dec.append("CONSERVADO")
        motivo.append("mandibula: conservada por apertura de boca (AU26/27)" if k in MANDIBULA else f"se mueve mas que el control (razon {razon_med[k]:.2f})")
T = pd.DataFrame(dict(punto=np.arange(u.N_LM), region=region, razon_rms_control=razon_med.round(3),
                      frac_sesiones_sobre_control=frac_sobre.round(2), decision=dec, motivo=motivo))
T.to_csv(os.path.join(a.salida, "tp_cara_03b_puntos.csv"), index=False)
keep = T.index[T["decision"] == "CONSERVADO"].tolist()
cols = [c for k in keep for c in (2 * k, 2 * k + 1)] + list(range(2 * u.N_LM, u.N_CANALES_TODOS))   # + cabeza (4) + yaw/pitch/roll (3)
json.dump(dict(puntos=keep, canales=cols, n_puntos=len(keep), n_canales=len(cols)),
          open(os.path.join(a.salida, "tp_cara_03b_puntos.json"), "w"))

p("\n" + "=" * 78 + "\nRESUMEN POR REGION (cuantos puntos se conservan)\n" + "=" * 78)
res = T.groupby("region").agg(puntos=("punto", "size"), conservados=("decision", lambda s: int((s == "CONSERVADO").sum())),
                              razon_mediana=("razon_rms_control", "median")).reset_index()
p(res.to_string(index=False))
p("\nExcluidos por regla:")
for s in ("S1", "S2", "S3"):
    p(f"  {s}: {int(T['motivo'].str.startswith(s).sum())}")
p(f"\nPuntos conservados: {len(keep)} de {u.N_LM} -> canales por persona: {len(cols)} ({2 * len(keep)} de puntos + 4 de cabeza + 3 de yaw/pitch/roll)")
borde = T[(T["decision"] != "EXCLUIDO") & (T["razon_rms_control"] < 1.3) & (T["region"] != "rigido(alineacion)")]
p(f"Puntos conservados con margen bajo (razon < 1.3, cerca de la regla S3): {len(borde)} -> revisa la lista en el CSV; "
  f"el analisis de sensibilidad es correr el modelado con 'todos' y con 'seleccion' y comparar.")
p("\nS3 excluyo, por region: " + "; ".join(f"{r_}: {int(((T['region'] == r_) & T['motivo'].str.startswith('S3')).sum())}" for r_ in sorted(set(region))))
p("\nLos 5 puntos con razon mas baja entre los conservados:")
p(T[T["decision"] == "CONSERVADO"].nsmallest(5, "razon_rms_control")[["punto", "region", "razon_rms_control"]].to_string(index=False))
log.guardar(os.path.join(a.salida, "tp_cara_03b_seleccionar_puntos.txt"))
