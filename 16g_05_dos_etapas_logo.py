"""
16g_05_dos_etapas_logo.py   (enfoque en dos etapas; equivale al script 09, con los 16 grupos)
Etapa 1: reconocer la accion de cada persona en cada segmento de 2 s desde sus keypoints de pose, cara y manos
         (acciones = las anotadas en speaker1-4; las que tienen menos de 40 ejemplos se agrupan en 'otra').
Etapa 2: puntaje de pareja = suma sobre acciones c (sin 'none' ni 'otra') de p_i(c) * p_j(c).
Validacion principal: UN GRUPO FUERA (entrena con 15 grupos, prueba con el 16o; 16 veces).
Validacion secundaria: por sesion (entrena con las A y prueba con las B, y viceversa), con la fuga entre personas que ello implica.

Corre las dos versiones del script 09: 'pose + manos' y 'pose sola' (se reportan ambas, como en la prueba de 2 grupos).
Con --sin-pose-sola corre solo 'pose + manos' (la mitad del tiempo).
Guarda las caracteristicas por persona la primera vez; si se corta, al volver a correr no las recalcula (--rehacer para forzar).

Uso (desde la carpeta scripts16g):   python 16g_05_dos_etapas_logo.py
Entradas: Root/GrupoNN/A|B/*.handsTracker.parquet, Root/mimicry_indicators.xlsx, salidas_16g/16g_01_series_<cod>.parquet, 16g_02_casos.parquet
Salidas:  salidas_16g/16g_05_caract_personas.parquet, 16g_05_resultados.csv, 16g_05_etapa1.csv,
          16g_05_puntajes_pareja.parquet, 16g_05_dos_etapas_logo.txt
Requiere: pandas, pyarrow, openpyxl, numpy, scikit-learn
"""
import glob, os, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
import util16g as u

warnings.filterwarnings("ignore")
MIN_CLASE = 40; N_BOOT = 500; BLOQUE = 10; SEMILLA = 0
ap = u.parser("Dos etapas, un grupo fuera")
ap.add_argument("--sin-pose-sola", action="store_true", help="correr solo pose + manos (mas rapido; omite la comparacion con pose sola)")
ap.add_argument("--rehacer", action="store_true", help="recalcular las caracteristicas por persona")
ap.add_argument("--arboles", type=int, default=300, help="arboles del Random Forest (300 como en la prueba de 2 grupos)")
a = ap.parse_args()
log = u.Log(); p = log.p
os.makedirs(a.salida, exist_ok=True)

ann = u.leer_anotaciones(a.anot)
ann["sesion"] = ann["Grupo"].str[:3]
ann["seg"] = (pd.to_timedelta(ann["time_start"]).dt.total_seconds() // 2).astype(int)
casos = pd.read_parquet(os.path.join(a.salida, "16g_02_casos.parquet"))
casos = casos[casos["usable"]].copy()
ses = sorted(casos["sesion"].unique())

def feats_pose(S):
    cc = [c for c in S.columns if c.startswith("c_")]
    pts = [0] + list(range(11, 23))
    cols_pos = [f"b{i}_{e}" for i in pts for e in "xy"]
    g = S.groupby(["track_id", "seg"])
    m = g[cols_pos + cc].mean().add_suffix("_m"); sd = g[cols_pos + cc].std().add_suffix("_sd")
    v = S.sort_values(["track_id", "frame"]).assign(**{c: S.sort_values(["track_id", "frame"]).groupby("track_id")[c].diff().abs() for c in cols_pos})
    v = v.groupby(["track_id", "seg"])[cols_pos].mean().add_suffix("_v")
    return pd.concat([m, sd, v], axis=1).reset_index()

def feats_manos(ruta, S):
    H = pd.read_parquet(ruta)
    base = H[["frame", "track_id", "hand"]].copy()
    mx = H["9_x"].values - H["0_x"].values; my = H["9_y"].values - H["0_y"].values
    esc = np.hypot(mx, my); esc = np.where(esc < 1e-6, np.nan, esc)
    for i in range(1, 21):
        base[f"{i}_x"] = (H[f"{i}_x"].values - H["0_x"].values) / esc
        base[f"{i}_y"] = (H[f"{i}_y"].values - H["0_y"].values) / esc
    base = base.merge(S[["frame", "track_id", "seg"]], on=["frame", "track_id"], how="inner")
    n_fr = S.groupby(["track_id", "seg"])["frame"].nunique().rename("n").reset_index()
    res = None
    for mano in ("Left", "Right"):
        b = base[base["hand"] == mano].drop(columns="hand")
        b = b.groupby(["frame", "track_id", "seg"]).mean().reset_index()
        g = b.groupby(["track_id", "seg"])
        f = g.mean().drop(columns="frame").add_prefix(f"h{mano[0]}_")
        f[f"h{mano[0]}_pres"] = g["frame"].nunique()
        f = f.reset_index().merge(n_fr, on=["track_id", "seg"], how="right")
        f[f"h{mano[0]}_pres"] = f[f"h{mano[0]}_pres"].fillna(0) / f["n"]
        f = f.drop(columns="n")
        res = f if res is None else res.merge(f, on=["track_id", "seg"], how="outer")
    return res

# ---------- caracteristicas por persona y segmento (se guardan para no repetir) ----------
ruta_f = os.path.join(a.salida, "16g_05_caract_personas.parquet")
if os.path.exists(ruta_f) and not a.rehacer:
    F = pd.read_parquet(ruta_f); hechas = set(F["sesion"].unique())
else:
    F = pd.DataFrame(); hechas = set()
pend = [s for s in ses if s not in hechas]
if pend:
    prog = u.Progreso(len(pend), "Caracteristicas por persona (pose, cara y manos): una sesion = una unidad", log)
    partes = [F] if len(F) else []
    for k, cod in enumerate(pend, 1):
        prog.actualizar(f"sesion {cod} ({k}/{len(pend)})")
        S = pd.read_parquet(os.path.join(a.salida, f"16g_01_series_{cod}.parquet"))
        fp = feats_pose(S)
        rutas = glob.glob(os.path.join(a.root, f"Grupo{cod[:2]}", cod[2], "*.handsTracker.parquet"))
        if not rutas:
            raise FileNotFoundError(f"falta handsTracker de {cod}: corre 16g_00_comprobar_datos.py")
        f = fp.merge(feats_manos(rutas[0], S), on=["track_id", "seg"], how="left"); f["sesion"] = cod
        partes.append(f); prog.actualizar(avance=1)
    prog.cerrar()
    F = pd.concat(partes, ignore_index=True); F.to_parquet(ruta_f, index=False)
F = F[F["sesion"].isin(ses)]

largo = []
for k in range(1, 5):
    d = ann[["sesion", "seg", f"speaker{k}"]].rename(columns={f"speaker{k}": "accion"}); d["track_id"] = k; largo.append(d)
Y = pd.concat(largo); Y["accion"] = Y["accion"].fillna("none").str.strip().str.lower()
D = F.merge(Y, on=["sesion", "seg", "track_id"], how="inner")
D["grupo"] = D["sesion"].str[:2]; D["letra"] = D["sesion"].str[2]
cuenta = D["accion"].value_counts()
raras = set(cuenta[cuenta < MIN_CLASE].index) - {"none"}
D["clase"] = D["accion"].where(~D["accion"].isin(raras), "otra")
FEAT = [c for c in D.columns if c.endswith(("_m", "_sd", "_v")) or c.startswith("h") and c[1] in "LR" and c[2] == "_"]
FEAT_POSE = [c for c in FEAT if not (c.startswith("hL_") or c.startswith("hR_"))]
p("Segmentos-persona con etiqueta:", len(D), "| grupos:", D["grupo"].nunique(), "| caracteristicas (pose+manos):", len(FEAT), "| solo pose:", len(FEAT_POSE))
p("Clases de accion (tras agrupar las raras en 'otra'):"); p(D["clase"].value_counts().to_string())

casos["letra"] = casos["sesion"].str[2]
casos["bloque"] = casos["sesion"] + "_" + (casos["seg"] // BLOQUE).astype(str)

def ic(y, s, blq):
    rng = np.random.default_rng(SEMILLA); ids = np.unique(blq); v = []
    por = {b: np.where(blq == b)[0] for b in ids}
    for _ in range(N_BOOT):
        sel = np.concatenate([por[b] for b in rng.choice(ids, len(ids))])
        if y[sel].sum(): v.append(average_precision_score(y[sel], s[sel]))
    return np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)

grupos = sorted(D["grupo"].unique())
PRUEBAS = [("logo", "grupo", [g for g in grupos], True)]
FOLDS = [("logo", "grupo", g, [x for x in grupos if x != g]) for g in grupos]
FOLDS += [("por_sesion", "letra", "B", ["A"]), ("por_sesion", "letra", "A", ["B"])]
CONJUNTOS = [("pose + manos", FEAT)] + ([] if a.sin_pose_sola else [("pose sola", FEAT_POSE)])

prog = u.Progreso(len(FOLDS) * len(CONJUNTOS), "Dos etapas: una unidad = una prueba (entrenar etapa 1 con el resto y puntuar el grupo)", log)
res = []; punt = []; e1 = []
for esq, col, te_v, tr_vs in FOLDS:
    for nombre_f, feat in CONJUNTOS:
        etiqueta = f"{col}={te_v}" if esq == "logo" else f"entrena {tr_vs[0]} -> prueba {te_v}"
        prog.actualizar(f"{etiqueta} | {nombre_f} | etapa 1: entrenando Random Forest")
        tr, te = D[D[col].isin(tr_vs)], D[D[col] == te_v].copy()
        modelo = make_pipeline(SimpleImputer(strategy="median"),
                               RandomForestClassifier(n_estimators=a.arboles, min_samples_leaf=3, class_weight="balanced_subsample",
                                                      n_jobs=-1, random_state=SEMILLA))
        modelo.fit(tr[feat], tr["clase"])
        P = pd.DataFrame(modelo.predict_proba(te[feat]), columns=modelo.classes_, index=te.index)
        pred = P.idxmax(axis=1); maj = tr["clase"].mode()[0]
        e1.append(dict(esquema=esq, prueba=etiqueta, caract=nombre_f, exactitud=accuracy_score(te["clase"], pred),
                       f1_macro=f1_score(te["clase"], pred, average="macro"), exactitud_clase_mayoritaria=(te["clase"] == maj).mean()))
        prog.actualizar(f"{etiqueta} | {nombre_f} | etapa 2: puntaje de pareja")
        usar = [c for c in P.columns if c not in ("none", "otra")]
        PP = P[usar].copy(); PP["sesion"] = te["sesion"].values; PP["seg"] = te["seg"].values; PP["track_id"] = te["track_id"].values
        ct = casos[casos[col] == te_v].copy()
        ct["i"] = ct["pareja"].str[0].astype(int); ct["j"] = ct["pareja"].str[2].astype(int)
        pi = ct.merge(PP, left_on=["sesion", "seg", "i"], right_on=["sesion", "seg", "track_id"], how="left")
        pj = ct.merge(PP, left_on=["sesion", "seg", "j"], right_on=["sesion", "seg", "track_id"], how="left")
        ct["puntaje"] = (pi[usar].values * pj[usar].values).sum(axis=1)
        ct["esquema"] = esq; ct["prueba"] = etiqueta; ct["caract"] = nombre_f
        punt.append(ct)
        for canal in ("cuerpo", "cara"):
            c = ct[ct["canal"] == canal].dropna(subset=["puntaje"])
            y, s, blq = c["y"].values, c["puntaje"].values, c["bloque"].values
            if y.sum() == 0:
                continue
            lo, hi = ic(y, s, blq)
            res.append(dict(esquema=esq, canal=canal, prueba=etiqueta, caract=nombre_f, modelo="dos_etapas", n=len(c), positivos=int(y.sum()),
                            prevalencia=y.mean(), pr_auc=average_precision_score(y, s), ic_inf=lo, ic_sup=hi, roc_auc=roc_auc_score(y, s)))
        prog.actualizar(avance=1)
prog.cerrar()

Res = pd.DataFrame(res); E1 = pd.DataFrame(e1)
Res.to_csv(os.path.join(a.salida, "16g_05_resultados.csv"), index=False)
E1.to_csv(os.path.join(a.salida, "16g_05_etapa1.csv"), index=False)
pd.concat(punt, ignore_index=True).to_parquet(os.path.join(a.salida, "16g_05_puntajes_pareja.parquet"), index=False)
pd.set_option("display.width", 250)
for nombre_f, _ in CONJUNTOS:
    p("\n" + "=" * 90); p(f"ETAPA 1 (reconocer la accion), {nombre_f}"); p("=" * 90)
    e = E1[(E1["caract"] == nombre_f) & (E1["esquema"] == "logo")]
    p(e[["prueba", "exactitud", "f1_macro", "exactitud_clase_mayoritaria"]].round(3).to_string(index=False))
    p(f"promedio un grupo fuera: exactitud={e['exactitud'].mean():.3f} | F1 macro={e['f1_macro'].mean():.3f}")
    r = Res[(Res["caract"] == nombre_f)].copy(); r["x_azar"] = r["pr_auc"] / r["prevalencia"]
    for esq in r["esquema"].unique():
        for canal in ("cuerpo", "cara"):
            d = r[(r["esquema"] == esq) & (r["canal"] == canal)]
            p(f"\nETAPA 2 [{esq} | {canal}] (x_azar = PR-AUC / prevalencia; meta >= 2)")
            p(d[["prueba", "positivos", "prevalencia", "pr_auc", "ic_inf", "ic_sup", "x_azar"]].round(3).to_string(index=False))
            p(f"promedio x_azar = {d['x_azar'].mean():.2f}")
log.guardar(os.path.join(a.salida, "16g_05_dos_etapas_logo.txt"))
