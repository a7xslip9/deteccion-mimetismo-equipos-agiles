"""
16g_01_preparar_series.py   (MS1: ingesta y limpieza; equivale al script 03 de la prueba de 2 grupos)
Convierte los archivos de cada sesion en series limpias por persona y cuadro. MISMOS parametros y pasos que el
script 03, aplicados a las 32 sesiones anotadas (se detectan solas desde el Excel de anotaciones):
 1. Puntos 0 a 22 de la pose.  2. Enmascara visibilidad < 0.5.  3. Quita saltos de munecas (> 100 px).
 4. Interpola huecos de hasta 5 cuadros.  5. Normaliza por los hombros.  6. Descriptores de cara (+ yaw/pitch/roll).
 7. Asigna cada cuadro a su segmento de 2 s.  8. Marca por persona y segmento si hay datos suficientes (>= 70 %).

Si el programa se corta, vuelve a ejecutarlo: se salta las sesiones que ya termino (usa --rehacer para forzar todo).
Si una sesion falla (archivo faltante o danado), lo anota y sigue con las demas; al final lista las que fallaron.

Uso (desde la carpeta scripts16g):
    python 16g_01_preparar_series.py
    python 16g_01_preparar_series.py --grupos 01 15        (prueba rapida con 2 grupos)
Entradas: Root/GrupoNN/A|B/*.parquet y .wav, Root/mimicry_indicators.xlsx
Salidas:  salidas_16g/16g_01_series_<cod>.parquet, 16g_01_cobertura_<cod>.parquet, 16g_01_preparar_series.txt
Requiere: pandas, pyarrow, numpy, openpyxl
"""
import glob, os, traceback, wave
import numpy as np, pandas as pd
import util16g as u

VIS_MIN = 0.5; SALTO_PX = 100; HUECO_MAX = 5; COBERTURA_MIN = 0.7
PUNTOS_OK = list(range(13, 23)); FRAC_PUNTOS_OK = 0.5; SEG_S = 2.0
PUNTOS = list(range(23)); MUNECAS = (15, 16)

ap = u.parser("MS1: series limpias de las 32 sesiones")
ap.add_argument("--rehacer", action="store_true", help="volver a procesar aunque la sesion ya este lista")
a = ap.parse_args()
os.makedirs(a.salida, exist_ok=True)
log = u.Log(); p = log.p


def buscar(c, suf, obligatorio=True):
    r = glob.glob(os.path.join(c, "*" + suf))
    if not r and obligatorio:
        raise FileNotFoundError(f"{suf} en {c}")
    return r[0] if r else None

def interp(df):
    return df.interpolate(method="linear", limit=HUECO_MAX, limit_area="inside")

OJO_D_EXT, OJO_I_EXT = 33, 263; BOCA_I, BOCA_D = 61, 291; LABIO_SUP, LABIO_INF = 13, 14
PARP = {"od": (159, 145), "oi": (386, 374)}; CEJA = {"cd": (105, 159), "ci": (334, 386)}

def d(df, i, j):
    return np.hypot(df[f"{i}_x"] - df[f"{j}_x"], df[f"{i}_y"] - df[f"{j}_y"])

def descriptores_cara(f):
    esc = d(f, OJO_D_EXT, OJO_I_EXT); r = pd.DataFrame(index=f.index)
    r["boca_apertura"] = d(f, LABIO_SUP, LABIO_INF) / esc; r["boca_ancho"] = d(f, BOCA_I, BOCA_D) / esc
    r["ojo_d"] = d(f, *PARP["od"]) / esc; r["ojo_i"] = d(f, *PARP["oi"]) / esc
    r["ceja_d"] = d(f, *CEJA["cd"]) / esc; r["ceja_i"] = d(f, *CEJA["ci"]) / esc
    return r

def procesar(cod):
    g, s = cod[:2], cod[2]
    c = os.path.join(a.root, f"Grupo{g}", s)
    pose = pd.read_parquet(buscar(c, ".poseTracker.parquet"))
    face = pd.read_parquet(buscar(c, ".faceTracker.parquet"))
    fdir_path = buscar(c, ".faceDirection.parquet", obligatorio=False)
    fdir = pd.read_parquet(fdir_path) if fdir_path else None
    if fdir is None:
        p(f"  {cod}: faceDirection no encontrado: sin yaw/pitch/roll")
    n_fr = int(pose["frame"].nunique())
    with wave.open(buscar(c, ".wav")) as w:
        dur = w.getnframes() / w.getframerate()
    fps = n_fr / dur
    frames = np.arange(1, n_fr + 1)
    filas = []
    est = {"mask_vis": [], "saltos": 0, "trans": 0, "nan_pose": [], "nan_cara": [], "cara_falta": []}
    for t in sorted(pose["track_id"].unique()):
        P = pose[pose["track_id"] == t].set_index("frame").reindex(frames)
        for i in PUNTOS:
            m = P[f"{i}_v"] < VIS_MIN
            est["mask_vis"].append(float(m.mean()))
            P.loc[m, [f"{i}_x", f"{i}_y"]] = np.nan
        for i in MUNECAS:
            dx = P[f"{i}_x"].diff(); dy = P[f"{i}_y"].diff()
            salto = np.hypot(dx, dy) > SALTO_PX
            est["saltos"] += int(salto.sum()); est["trans"] += int(dx.notna().sum())
            P.loc[salto, [f"{i}_x", f"{i}_y"]] = np.nan
        cols = [f"{i}_{e}" for i in PUNTOS for e in "xy"]
        X = interp(P[cols])
        est["nan_pose"].append(float(X.isna().mean().mean()))
        cx = (X["11_x"] + X["12_x"]) / 2; cy = (X["11_y"] + X["12_y"]) / 2
        ancho = np.hypot(X["11_x"] - X["12_x"], X["11_y"] - X["12_y"]).median()
        for i in PUNTOS:
            X[f"{i}_x"] = (X[f"{i}_x"] - cx) / ancho; X[f"{i}_y"] = (X[f"{i}_y"] - cy) / ancho
        X.columns = ["b" + c_ for c_ in X.columns]
        F = face[face["track_id"] == t].set_index("frame").reindex(frames)
        est["cara_falta"].append(float(F["0_x"].isna().mean()))
        D = interp(descriptores_cara(F))
        if fdir is not None:
            Q = fdir[fdir["track_id"] == t].set_index("frame").reindex(frames)[["yaw", "pitch", "roll"]]
            D = pd.concat([D, interp(Q)], axis=1)
        est["nan_cara"].append(float(D.isna().mean().mean()))
        R = pd.concat([X, D.add_prefix("c_")], axis=1)
        R.insert(0, "track_id", t); R.insert(1, "t", (frames - 1) / fps)
        R.insert(2, "seg", np.floor((frames - 1) / fps / SEG_S).astype(int))
        R.index.name = "frame"; filas.append(R.reset_index())
    S = pd.concat(filas, ignore_index=True)
    cc = [x for x in S.columns if x.startswith("c_")]
    cok = [f"b{i}_{e}" for i in PUNTOS_OK for e in "xy"]
    S["_ok_b"] = S[cok].notna().mean(axis=1) >= FRAC_PUNTOS_OK
    S["_ok_c"] = S[cc].notna().all(axis=1)
    cov = S.groupby(["track_id", "seg"])[["_ok_b", "_ok_c"]].mean()
    cov["cuerpo_ok"] = cov["_ok_b"] >= COBERTURA_MIN; cov["cara_ok"] = cov["_ok_c"] >= COBERTURA_MIN
    S = S.drop(columns=["_ok_b", "_ok_c"])
    S.to_parquet(os.path.join(a.salida, f"16g_01_series_{cod}.parquet"), index=False)
    cov.reset_index().to_parquet(os.path.join(a.salida, f"16g_01_cobertura_{cod}.parquet"), index=False)
    p(f"  {cod}: cuadros={n_fr}, duracion={dur:.0f} s, fps={fps:.2f}, segmentos={int(S['seg'].max()) + 1} | "
      f"visibilidad<{VIS_MIN}: {np.mean(est['mask_vis']) * 100:.1f} % | saltos muneca: {est['saltos'] / max(est['trans'], 1) * 100:.2f} % | "
      f"cara sin malla: {np.mean(est['cara_falta']) * 100:.1f} %")


ann = u.leer_anotaciones(a.anot)
ses = u.sesiones_anotadas(ann, a.grupos)
pend = [s for s in ses if a.rehacer or not os.path.exists(os.path.join(a.salida, f"16g_01_cobertura_{s}.parquet"))]
log.lineas.append(f"--- ejecucion: {len(ses)} sesiones, {len(pend)} por procesar ---")
print(f"Sesiones anotadas: {len(ses)} | ya listas: {len(ses) - len(pend)} | por procesar: {len(pend)}")
fallas = []
if pend:
    prog = u.Progreso(len(pend), "MS1: preparando series (una sesion = una unidad)", log)
    for k, cod in enumerate(pend, 1):
        prog.actualizar(f"sesion {cod} ({k}/{len(pend)})")
        try:
            procesar(cod)
        except Exception as e:
            fallas.append(cod); p(f"  {cod}: FALLO -> {type(e).__name__}: {e}")
            log.lineas.append(traceback.format_exc())
        prog.actualizar(avance=1)
    prog.cerrar()
listas = [s for s in ses if os.path.exists(os.path.join(a.salida, f"16g_01_cobertura_{s}.parquet"))]
p(f"\nSesiones listas: {len(listas)} de {len(ses)}")
if fallas:
    p("Sesiones que fallaron:", " ".join(fallas), "(revisa los archivos con 16g_00_comprobar_datos.py y vuelve a correr este script)")
log.guardar(os.path.join(a.salida, "16g_01_preparar_series.txt"), anexar=True)
