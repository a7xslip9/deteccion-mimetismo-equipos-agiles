"""
tp_cara_01_entender_datos.py        (CRISP-DM, fase 2: COMPRENSION DE LOS DATOS; canal CARA)
Antes de decidir como procesar la cara, mide como son realmente los datos. No modifica nada.
Responde: que trae el parquet de cara, que resolucion tiene, cuantos cuadros faltan, cuantos positivos hay por grupo,
y si las anotaciones y los archivos son coherentes. Los numeros que salen aqui son la base de varias decisiones
(D-06, D-07, D-08, D-09 en DECISIONES_cara.md): por eso se guardan.

Que mide:
  1. Entorno (versiones de las librerias, nucleos, GPU).
  2. Anotaciones: sesiones, filas, grupos, positivos de mimetismo FACIAL por grupo y por pareja (celda face_mimicry).
  3. Por sesion: archivos presentes (faceTracker, faceDirection, wav), cuadros, fps, personas, % de cuadros con malla facial.
  4. Resolucion de la malla: distancia entre ojos (px), apertura de boca (px), si las coordenadas son enteras, cuanto
     cambia un punto fijo de un cuadro a otro (ruido de cuantizacion), rango y valores distintos de z.

Uso (desde la carpeta scripts_tp):
    python tp_cara_01_entender_datos.py
    python tp_cara_01_entender_datos.py --grupos 01 15        (prueba rapida)
Entradas: Root/GrupoNN/A|B/*.faceTracker.parquet (solo algunas columnas), .poseTracker.parquet, .wav, Root/mimicry_indicators.xlsx
Salidas:  salidas_tp/tp_cara_01_sesiones.csv, tp_cara_01_positivos.csv, tp_cara_01_entender_datos.txt
"""
import importlib, itertools, os, platform, re, sys, traceback
import numpy as np, pandas as pd
import tp_util as u

ap = u.parser("Comprension de los datos de cara")
a = ap.parse_args()
os.makedirs(a.salida, exist_ok=True)
log = u.Log(); p = log.p

p("=" * 78); p("1. ENTORNO"); p("=" * 78)
p("Sistema:", platform.platform(), "| nucleos logicos:", os.cpu_count(), "| Python:", sys.version.split()[0])
for lib in ["numpy", "pandas", "pyarrow", "openpyxl", "scipy", "sklearn", "aeon", "torch", "matplotlib"]:
    try:
        m = importlib.import_module(lib); p(f"{lib}: {m.__version__}")
    except Exception:
        p(f"{lib}: NO INSTALADO")
try:
    import torch
    p("GPU (torch.cuda):", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no disponible (el Transformer correra en CPU, mucho mas lento)")
except Exception:
    pass

# ------------------------------------------------------------------------------------------------ anotaciones
p("\n" + "=" * 78); p("2. ANOTACIONES"); p("=" * 78)
ann = u.leer_anotaciones(a.anot)
ses = u.sesiones_anotadas(ann, a.grupos)
ann = ann[ann["Grupo"].str[:3].isin(ses)].copy()
PAT = re.compile(r"([1-4])\s*(=|->)\s*([1-4])")
PAREJAS = list(itertools.combinations([1, 2, 3, 4], 2))
p(f"Sesiones anotadas: {len(ses)} | grupos: {len({s[:2] for s in ses})} | filas (segmentos de 2 s): {len(ann)}")
t = pd.to_timedelta(ann["time_start"]).dt.total_seconds()
p(f"Los inicios de segmento son multiplos de 2 s: {bool((t % 2 == 0).all())}")
filas = []
for r in ann.itertuples():
    cod = r.Grupo[:3]
    celda = "" if pd.isna(r.face_mimicry) else str(r.face_mimicry)
    ev = [(tuple(sorted((int(m.group(1)), int(m.group(3))))), "sync" if m.group(2) == "=" else "async") for m in PAT.finditer(celda)]
    for pr in PAREJAS:
        tipos = [tp for (q, tp) in ev if q == pr]
        filas.append((cod, cod[:2], f"{pr[0]}-{pr[1]}", int(len(tipos) > 0), int("sync" in tipos), int("async" in tipos)))
E = pd.DataFrame(filas, columns=["sesion", "grupo", "pareja", "y", "sync", "async"])
tot = E["y"].shape[0]
p(f"Casos pareja x segmento (cara): {tot} | positivos: {int(E['y'].sum())} ({E['y'].mean() * 100:.2f} %) "
  f"| sincronos: {int(E['sync'].sum())} | asincronos: {int(E['async'].sum())}")
celdas_raras = ann[~ann["face_mimicry"].fillna("0").astype(str).str.match(r"^(0|\s*[1-4]\s*(=|->)\s*[1-4]\s*(,\s*[1-4]\s*(=|->)\s*[1-4]\s*)*)$")]
p(f"Celdas face_mimicry que NO calzan con el formato esperado ('0' o 'i=j', 'i->j' separados por coma): {len(celdas_raras)}")
if len(celdas_raras):
    p("  ejemplos:", celdas_raras["face_mimicry"].value_counts().head(5).to_dict())
G = E.groupby("grupo").agg(casos=("y", "size"), positivos=("y", "sum")); G["prevalencia_%"] = (G["positivos"] / G["casos"] * 100).round(2)
p("\nPositivos de mimetismo facial por grupo (antes de descartar datos invalidos):"); p(G.to_string())
P = E.groupby("pareja").agg(casos=("y", "size"), positivos=("y", "sum")); P["prevalencia_%"] = (P["positivos"] / P["casos"] * 100).round(2)
p("\nPor pareja de personas (1 a 4 = speaker1 a speaker4):"); p(P.to_string())
if (G["positivos"] < 10).any():
    p("AVISO: grupos con menos de 10 positivos (resultado por grupo muy inestable):", list(G.index[G["positivos"] < 10]))
G.reset_index().to_csv(os.path.join(a.salida, "tp_cara_01_positivos.csv"), index=False)

# ------------------------------------------------------------------------------------------------ por sesion
p("\n" + "=" * 78); p("3. ARCHIVOS Y COBERTURA POR SESION"); p("=" * 78)
COLS = ["frame", "track_id", "x1", "y1", "x2", "y2"] + [f"{k}_{e}" for k in (0, 1, 13, 14, 33, 61, 263, 291) for e in "xy"] + ["1_z"]
filas = []; fallas = []
prog = u.Progreso(len(ses), "Midiendo cada sesion", log)
for cod in ses:
    prog.actualizar(f"sesion {cod}")
    try:
        c = u.carpeta_sesion(a.root, cod)
        n_fr, dur, fps = u.fps_sesion(c)
        F = pd.read_parquet(u.buscar(c, ".faceTracker.parquet"), columns=COLS)
        tiene_dir = u.buscar(c, ".faceDirection.parquet", obligatorio=False) is not None
        pers = sorted(F["track_id"].unique())
        esperado = n_fr * len(pers)
        d_ojos = np.hypot(F["33_x"] - F["263_x"], F["33_y"] - F["263_y"])
        ap_boca = (F["13_y"] - F["14_y"]).abs()
        nariz = F[F["track_id"] == pers[0]].sort_values("frame")["1_x"].diff().abs()
        filas.append(dict(sesion=cod, cuadros=n_fr, duracion_s=round(dur, 1), fps=round(fps, 2), personas=len(pers),
                          filas_cara=len(F), cobertura_malla=round(len(F) / esperado, 4), faceDirection=tiene_dir,
                          coords_enteras=bool(np.allclose(F["0_x"], F["0_x"].round()) and np.allclose(F["0_y"], F["0_y"].round())),
                          dist_ojos_px_mediana=round(float(d_ojos.median()), 1), dist_ojos_px_p5=round(float(d_ojos.quantile(.05)), 1),
                          apertura_boca_px_mediana=float(ap_boca.median()), apertura_boca_px_p95=float(ap_boca.quantile(.95)),
                          nariz_dx_cero_frac=round(float((nariz == 0).mean()), 3), nariz_dx_mediana_px=float(nariz.median()),
                          z_valores_distintos=int(F["1_z"].nunique()), ancho_bbox_px_mediana=float((F["x2"] - F["x1"]).median())))
    except Exception as e:
        fallas.append(cod); p(f"  {cod}: FALLO -> {type(e).__name__}: {e}"); log.lineas.append(traceback.format_exc())
    prog.actualizar(avance=1)
prog.cerrar()
S = pd.DataFrame(filas)
S.to_csv(os.path.join(a.salida, "tp_cara_01_sesiones.csv"), index=False)
if len(S):
    p(S[["sesion", "cuadros", "fps", "personas", "cobertura_malla", "faceDirection", "dist_ojos_px_mediana", "apertura_boca_px_mediana"]].to_string(index=False))
    p("\n" + "=" * 78); p("4. RESOLUCION DE LA MALLA (lo que limita el uso de 'todos los puntos')"); p("=" * 78)
    p(f"Coordenadas enteras (pixeles) en todas las sesiones: {bool(S['coords_enteras'].all())}")
    p(f"Distancia entre ojos: mediana {S['dist_ojos_px_mediana'].median():.0f} px (rango entre sesiones {S['dist_ojos_px_mediana'].min():.0f} a {S['dist_ojos_px_mediana'].max():.0f})")
    p(f"Apertura de boca: mediana {S['apertura_boca_px_mediana'].median():.1f} px, percentil 95 {S['apertura_boca_px_p95'].median():.1f} px")
    p(f"Un punto fijo (punta de la nariz) cambia en promedio {S['nariz_dx_mediana_px'].median():.1f} px entre cuadros; "
      f"no cambia en el {S['nariz_dx_cero_frac'].median() * 100:.0f} % de los cuadros")
    p(f"Valores distintos de z en la punta de la nariz (por sesion, mediana): {S['z_valores_distintos'].median():.0f}")
    q = 0.5 / S["dist_ojos_px_mediana"].median()
    p(f"\nLECTURA: el redondeo a pixel entero introduce un error de hasta +-0,5 px = {q * 100:.1f} % de la distancia entre ojos en cada coordenada.")
    p("Los movimientos faciales sutiles (apertura de boca de 1 px) son del mismo orden que ese error. Consecuencias de diseno:")
    p("  - se suaviza en el tiempo (D-07) y se alinea con muchos puntos (D-08), que promedian el ruido;")
    p("  - NO se estandariza cada punto por su propia desviacion (amplificaria el ruido de los puntos casi inmoviles, D-10);")
    p("  - z se descarta: tiene muy pocos valores distintos (D-02).")
    if not S["faceDirection"].all():
        p("AVISO: sin faceDirection (yaw/pitch/roll) en:", " ".join(S.loc[~S["faceDirection"], "sesion"]), "-> esos tres canales quedaran en 0 en esas sesiones.")
    bajos = S[S["cobertura_malla"] < 0.9]
    if len(bajos):
        p("AVISO: sesiones con menos de 90 % de cuadros con malla facial:", " ".join(bajos["sesion"]))
if fallas:
    p("\nSESIONES CON FALLO:", " ".join(fallas), "(revisa los archivos y vuelve a correr)")
log.guardar(os.path.join(a.salida, "tp_cara_01_entender_datos.txt"))
