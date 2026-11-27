"""
tp_cuerpo_01_entender_datos.py        (CRISP-DM, fase 2: COMPRENSION DE LOS DATOS; canal CUERPO)
No modifica nada. Mide lo que hay que saber ANTES de modelar el cuerpo y lo deja por escrito:
  1. Entorno (versiones, GPU).
  2. Anotaciones: cuantos casos y positivos de mimetismo CORPORAL (columna body_mimicry) hay por grupo y por pareja, y
     las acciones anotadas por persona (23 etiquetas). Las acciones NO se usan como entrada de ningun modelo (B-01).
  3. Archivos por sesion: pose (33 puntos), manos (21 puntos por mano), cobertura, duplicados, resolucion.
  4. Visibilidad por punto de la pose (base de la seleccion de puntos, B-07) y deteccion de manos.

Uso (desde la carpeta scripts_tp, con la carpeta Root al lado):
    python tp_cuerpo_01_entender_datos.py
    python tp_cuerpo_01_entender_datos.py --grupos 01 15        (prueba rapida)
Salidas (carpeta salidas_tp_cuerpo): tp_cuerpo_01_sesiones.csv, tp_cuerpo_01_positivos.csv, tp_cuerpo_01_acciones.csv,
          tp_cuerpo_01_visibilidad_pose.csv, tp_cuerpo_01_entender_datos.txt
"""
import os, re, platform, warnings, traceback
import numpy as np, pandas as pd
import tp_util as u

warnings.filterwarnings("ignore")
ap = u.parser("Comprension de los datos de cuerpo", salida=u.SALIDA_CUERPO)
a = ap.parse_args()
os.makedirs(a.salida, exist_ok=True)
log = u.Log(); p = log.p
linea = lambda t: p("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

linea("1. ENTORNO")
p(f"Sistema: {platform.platform()} | nucleos logicos: {os.cpu_count()} | Python: {platform.python_version()}")
for m in ("numpy", "pandas", "pyarrow", "openpyxl", "scipy", "sklearn", "aeon", "torch", "matplotlib"):
    try:
        mod = __import__(m); p(f"{m}: {mod.__version__}")
    except Exception as e:
        p(f"{m}: NO DISPONIBLE ({type(e).__name__})")
try:
    import torch
    p("GPU (torch.cuda):", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no disponible")
except Exception:
    pass

linea("2. ANOTACIONES")
ann = u.leer_anotaciones(a.anot)
if "body_mimicry" not in ann.columns:
    raise SystemExit(f"No existe la columna body_mimicry en el Excel. Columnas: {list(ann.columns)}")
ann["cod"] = ann["Grupo"].astype(str).str[:3]; ann["grupo"] = ann["cod"].str[:2]
ses = u.sesiones_anotadas(ann, a.grupos)
ann = ann[ann["cod"].isin(ses)].copy()
PAT = re.compile(r"([1-4])\s*(=|->)\s*([1-4])")
filas, mal = [], 0
for r in ann.itertuples():
    c = "" if pd.isna(r.body_mimicry) else str(r.body_mimicry).strip()
    ev = [(tuple(sorted((int(m.group(1)), int(m.group(3))))), m.group(2)) for m in PAT.finditer(c)]
    if c not in ("", "0", "nan") and not ev:
        mal += 1
    for pr in {e[0] for e in ev}:
        filas.append(dict(grupo=r.grupo, pareja=f"{pr[0]}-{pr[1]}", sync=int(any(e[0] == pr and e[1] == "=" for e in ev)),
                          asyn=int(any(e[0] == pr and e[1] == "->" for e in ev))))
E = pd.DataFrame(filas)
n_seg = len(ann); n_casos = n_seg * 6
p(f"Sesiones anotadas: {len(ses)} | grupos: {ann['grupo'].nunique()} | filas (segmentos de 2 s): {n_seg}")
p(f"Casos pareja x segmento (cuerpo): {n_casos} | positivos: {len(E)} ({len(E) / n_casos * 100:.2f} %) | sincronos: {int(E['sync'].sum())} | asincronos: {int(E['asyn'].sum())}")
p(f"Celdas body_mimicry que NO calzan con el formato esperado: {mal}")
Gp = E.groupby("grupo").size().reindex(sorted(ann["grupo"].unique()), fill_value=0).rename("positivos").to_frame()
Gp.insert(0, "casos", ann.groupby("grupo").size() * 6); Gp["prevalencia_%"] = (Gp["positivos"] / Gp["casos"] * 100).round(2)
Gp.index.name = "grupo"; Gp.to_csv(os.path.join(a.salida, "tp_cuerpo_01_positivos.csv"))
p("\nPositivos de mimetismo corporal por grupo (antes de descartar datos invalidos):"); p(Gp.to_string())
Pp = E.groupby("pareja").size().rename("positivos").to_frame(); Pp.insert(0, "casos", n_seg); Pp["prevalencia_%"] = (Pp["positivos"] / Pp["casos"] * 100).round(2)
p("\nPor pareja de personas (1 a 4 = speaker1 a speaker4):"); p(Pp.to_string())
sin = list(Gp.index[Gp["positivos"] == 0]); pocos = list(Gp.index[(Gp["positivos"] > 0) & (Gp["positivos"] < 10)])
if sin: p("AVISO: grupos sin positivos (no sirven de prueba):", sin)
if pocos: p("AVISO: grupos con menos de 10 positivos (resultado individual inestable):", pocos)

acc = pd.concat([ann[c] for c in ("speaker1", "speaker2", "speaker3", "speaker4") if c in ann.columns]).astype(str)
acc = acc.str.split(",").explode().str.strip().str.lower()
AC = acc.value_counts().rename_axis("accion").reset_index(name="registros"); AC.to_csv(os.path.join(a.salida, "tp_cuerpo_01_acciones.csv"), index=False)
p(f"\nAcciones anotadas por persona (speaker1..4): {len(AC)} etiquetas distintas (incluye 'none'); las 10 mas frecuentes:")
p(AC.head(10).to_string(index=False))
p("NOTA: estas acciones son anotacion manual. No se usan como entrada de los modelos (B-01); solo describen el dataset.")

linea("3. ARCHIVOS Y COBERTURA POR SESION")
filas, vis_all = [], []
for cod in ses:
    try:
        c = u.carpeta_sesion(a.root, cod)
        n_fr, dur, fps = u.fps_sesion(c)
        P = pd.read_parquet(u.buscar(c, ".poseTracker.parquet"))
        rh = u.buscar(c, ".handsTracker.parquet", obligatorio=False)
        H = pd.read_parquet(rh) if rh else None
        pres = P.groupby("track_id").size()
        V = np.stack([P[f"{k}_v"].to_numpy(float) for k in range(u.N_POSE)], 1)
        vis = (V >= u.UMBRAL_VIS).mean(0)
        vis_all.append(pd.Series(vis, name=cod))
        X = np.stack([P[f"{k}_x"].to_numpy(float) for k in range(u.N_POSE)], 1); Y = np.stack([P[f"{k}_y"].to_numpy(float) for k in range(u.N_POSE)], 1)
        ancho = np.hypot(X[:, 11] - X[:, 12], Y[:, 11] - Y[:, 12])
        sh_ok = (V[:, 11] >= u.UMBRAL_VIS) & (V[:, 12] >= u.UMBRAL_VIS)
        fila = dict(sesion=cod, cuadros=n_fr, duracion_s=round(dur, 1), fps=round(fps, 2), personas=int(P["track_id"].nunique()),
                    filas_pose=len(P), cobertura_hombros=round(float(sh_ok.mean()), 4),
                    coords_enteras=bool(np.all(X == np.round(X)) and np.all(Y == np.round(Y))),
                    ancho_hombros_px_mediana=round(float(np.nanmedian(ancho[sh_ok])), 1) if sh_ok.any() else np.nan,
                    vis_superior=round(float(vis[:23].mean()), 3), vis_caderas=round(float(vis[23:25].mean()), 3), vis_piernas=round(float(vis[25:].mean()), 3))
        if H is not None:
            dup = int(H.duplicated(["frame", "track_id", "hand"]).sum())
            Hu = H.drop_duplicates(["frame", "track_id", "hand"])
            tam = np.hypot(Hu["0_x"] - Hu["9_x"], Hu["0_y"] - Hu["9_y"])
            nper = max(int(P["track_id"].nunique()), 1)
            fila.update(manos_filas=len(H), manos_duplicadas=dup, mano_I_frac=round(float((Hu["hand"] == "Left").sum() / (n_fr * nper)), 3),
                        mano_D_frac=round(float((Hu["hand"] == "Right").sum() / (n_fr * nper)), 3), mano_tam_px_mediana=round(float(np.median(tam)), 1),
                        mano_vis_constante=bool(H["0_v"].nunique() == 1))
        else:
            fila.update(manos_filas=0)
        filas.append(fila)
    except Exception as e:
        p(f"  {cod}: FALLO -> {type(e).__name__}: {e}"); log.lineas.append(traceback.format_exc())
S = pd.DataFrame(filas); S.to_csv(os.path.join(a.salida, "tp_cuerpo_01_sesiones.csv"), index=False)
p(S[["sesion", "cuadros", "fps", "personas", "cobertura_hombros", "ancho_hombros_px_mediana", "vis_caderas", "vis_piernas"]
    + [c for c in ("mano_I_frac", "mano_D_frac", "mano_tam_px_mediana", "manos_duplicadas") if c in S.columns]].to_string(index=False))
if "manos_filas" in S and (S["manos_filas"] == 0).any():
    p("AVISO: sesiones SIN archivo de manos:", " ".join(S.loc[S["manos_filas"] == 0, "sesion"]))
baja = S.loc[S["cobertura_hombros"] < 0.9, "sesion"].tolist()
if baja: p("AVISO: sesiones con menos de 90 % de cuadros con ambos hombros visibles:", " ".join(baja))

linea("4. VISIBILIDAD POR PUNTO DE LA POSE (fraccion de cuadros con visibilidad >= 0,5, promedio de sesiones)")
VIS = pd.concat(vis_all, axis=1).T if vis_all else pd.DataFrame()
if len(VIS):
    VIS.columns = [f"p{k}" for k in range(u.N_POSE)]; VIS.index.name = "sesion"; VIS.round(3).to_csv(os.path.join(a.salida, "tp_cuerpo_01_visibilidad_pose.csv"))
    m = VIS.mean(0)
    p("Puntos 0-10 (cabeza):", f"{m[:11].min():.2f} a {m[:11].max():.2f}")
    p("Puntos 11-22 (hombros, codos, munecas, dedos):", " ".join(f"{k}:{m.iloc[k]:.2f}" for k in range(11, 23)))
    p("Puntos 23-24 (caderas):", f"{m.iloc[23]:.2f} y {m.iloc[24]:.2f}")
    p("Puntos 25-32 (piernas):", f"max {m.iloc[25:].max():.2f}")
    p(f"Puntos con visibilidad media < {0.70:.2f}: {[int(c[1:]) for c in m.index[m < 0.70]]}")
if len(S):
    p(f"\nCoordenadas enteras en todas las sesiones: {bool(S['coords_enteras'].all())}")
    p(f"Ancho de hombros: mediana {S['ancho_hombros_px_mediana'].median():.0f} px (rango {S['ancho_hombros_px_mediana'].min():.0f} a {S['ancho_hombros_px_mediana'].max():.0f}). "
      f"El error de redondeo a pixel (+-0,5 px) es ~{0.5 / S['ancho_hombros_px_mediana'].median() * 100:.2f} % del ancho de hombros: mucho menor que en la cara.")
    if "mano_tam_px_mediana" in S:
        p(f"Tamano de la mano (muneca a base del dedo medio): mediana {S['mano_tam_px_mediana'].median():.0f} px; el redondeo equivale a ~{0.5 / S['mano_tam_px_mediana'].median() * 100:.1f} % de ese tamano.")
log.guardar(os.path.join(a.salida, "tp_cuerpo_01_entender_datos.txt"))
