"""
16g_12_modelo_final.py   (modelo final: etapa 1 entrenada con los 16 grupos y datos sinteticos, y puntaje de pares)

Que es: la configuracion que cumplio el criterio en la validacion dejando un grupo fuera (16g_10: representacion base de 178
caracteristicas, 3.000 ejemplos por accion de las acciones corporales, hasta 40 variantes por ejemplo real, 300 arboles, semilla 0).
Aqui se entrena UNA sola vez con los 16 grupos anotados (mas los sinteticos generados con los 16 grupos) y se guarda en un archivo.

Que NO es: una evaluacion. El modelo final ya vio los 16 grupos, asi que su desempeno sobre esos mismos grupos es optimista y no se
reporta como resultado. El desempeno que se atribuye a este modelo es el de la validacion dejando un grupo fuera (16g_10_base_resultados.csv:
cuerpo ~2,0 veces el azar, cara ~6 veces el azar); este script lo copia en la ficha del modelo.
Un mismo modelo sirve a los dos canales: el puntaje del par es el mismo, y se contrasta con la anotacion corporal o facial.

Modos:
  python 16g_12_modelo_final.py                                 entrena, guarda el modelo y puntua los pares de los 16 grupos
  python 16g_12_modelo_final.py --modo puntuar --grupos 01 15   usa el modelo guardado para puntuar los pares de esos grupos
Opciones del entrenamiento: --objetivos 3000   --max-por-ejemplo 40   --reserva 5000   --arboles 300   --semilla 0   --rep base
(la reserva de sinteticos se genera como en la validacion --objetivos 3000 5000 --max-por-ejemplo 40, y de ella se sortean los 3000 por accion)
Entradas: las del 16g_10 (carpeta Root, salidas_16g con 16g_01, 16g_02, 16g_05 y 16g_08); para puntuar, solo 16g_02, 16g_05 y el modelo.
Salidas:  salidas_16g/16g_12_modelo_final.joblib (modelo + lista de caracteristicas + configuracion),
          16g_12_ficha_modelo.txt (que contiene, con que datos, que desempeno validado), 16g_12_puntajes_pares.parquet (puntajes, en muestra),
          16g_12_candidatos.csv (los 25 segmentos-par con mayor puntaje por sesion: lista para revision de quien anota), 16g_12_sinteticos.csv, 16g_12_fidelidad.csv
Limite: puntuar sesiones nuevas requiere haber calculado antes sus caracteristicas (16g_00 a 16g_02 y la extraccion de caracteristicas del 16g_05).
Requiere: pandas, pyarrow, openpyxl, numpy, scikit-learn, joblib
"""
import glob, os, sys, json, datetime, warnings
import joblib
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
import util16g as u

warnings.filterwarnings("ignore")
MIN_CLASE = 40; N_BOOT = 500; BLOQUE = 10; META = 2.0
ROT_POSE, ROT_MANO, ESC_MIN, ESC_MAX, VEL_MIN, VEL_MAX, RUIDO = 10.0, 15.0, 0.9, 1.1, 0.85, 1.15, 0.5
CARA_POR_DEFECTO = ["hand in mouth", "lean backwards", "hand in cheek", "hand in chin", "hand in head", "fix hair"]
CUERPO_POR_DEFECTO = ["hand gesture", "playing with pen", "holding hand", "interlocking fingers", "fiddling hands",
                      "interlocked fingers", "writing", "three fingers", "four fingers", "five fingers"]

ap = u.parser("Datos sinteticos, canal corporal, un grupo fuera")
ap.add_argument("--objetivos", nargs="+", type=int, default=[3000], help="ejemplos por accion hasta los que se completa (el modelo final usa un solo valor)")
ap.add_argument("--max-por-ejemplo", type=int, default=40, help="maximo de variantes sinteticas por ejemplo real")
ap.add_argument("--arboles", type=int, default=300)
ap.add_argument("--semilla", type=int, default=0)
ap.add_argument("--reserva", type=int, default=5000, help="la reserva de sinteticos se genera como para esta cantidad (5000 en la validacion del 16g_10, de la que se sortean los 3000)")
ap.add_argument("--modo", choices=["entrenar", "puntuar"], default="entrenar", help="entrenar: entrena y guarda el modelo; puntuar: usa el modelo guardado")
ap.add_argument("--rep", choices=["base", "fina"], default="base", help="base: caracteristicas del 16g_05; fina: se agregan las caracteristicas finas de las manos")
ap.add_argument("--aumentar", choices=["cuerpo", "cara", "ambos"], default="cuerpo",
                help="acciones que se aumentan: las del canal corporal (por defecto, decision del informe), las del facial (control) o ambas")
a = ap.parse_args()
log = u.Log(); p = log.p
PRE = "16g_12_"


def puntuar_pares(M, F, casos):
    """Puntaje de cada par y segmento: suma sobre las acciones (sin none ni otra) de p_i(c) * p_j(c), con p la probabilidad de la etapa 1."""
    modelo, FEAT_M = M["modelo"], M["caracteristicas"]
    P = pd.DataFrame(modelo.predict_proba(F[FEAT_M]), columns=modelo.classes_, index=F.index)
    usar = [c for c in P.columns if c not in ("none", "otra")]
    PP = P[usar].copy(); PP["sesion"] = F["sesion"].values; PP["seg"] = F["seg"].values; PP["track_id"] = F["track_id"].values
    ct = casos.copy(); ct["i"] = ct["pareja"].str[0].astype(int); ct["j"] = ct["pareja"].str[2].astype(int)
    pi = ct.merge(PP, left_on=["sesion", "seg", "i"], right_on=["sesion", "seg", "track_id"], how="left")
    pj = ct.merge(PP, left_on=["sesion", "seg", "j"], right_on=["sesion", "seg", "track_id"], how="left")
    ct["puntaje"] = (pi[usar].values * pj[usar].values).sum(axis=1)
    return ct

def candidatos(ct, n=25):
    d = ct.dropna(subset=["puntaje"]).drop_duplicates(["sesion", "seg", "pareja"])
    return d.sort_values("puntaje", ascending=False).groupby("sesion").head(n).sort_values(["sesion", "puntaje"], ascending=[True, False])[["sesion", "seg", "pareja", "puntaje"]]

if a.modo == "puntuar":
    ruta = os.path.join(a.salida, "16g_12_modelo_final.joblib")
    if not os.path.exists(ruta):
        raise SystemExit("No existe el modelo guardado: primero se entrena con  python 16g_12_modelo_final.py")
    M = joblib.load(ruta)
    F = pd.read_parquet(os.path.join(a.salida, "16g_05_caract_personas.parquet"))
    cs = pd.read_parquet(os.path.join(a.salida, "16g_02_casos.parquet")); cs = cs[cs["usable"]].copy()
    if a.grupos:
        cs = cs[cs["sesion"].str[:2].isin(a.grupos)]; F = F[F["sesion"].str[:2].isin(a.grupos)]
    ct = puntuar_pares(M, F, cs)
    ct[["sesion", "seg", "pareja", "canal", "puntaje"]].to_parquet(os.path.join(a.salida, "16g_12_puntajes_pares_puntuar.parquet"), index=False)
    candidatos(ct).to_csv(os.path.join(a.salida, "16g_12_candidatos_puntuar.csv"), index=False)
    print(f"Pares puntuados: {len(ct)} en {ct['sesion'].nunique()} sesiones. Salidas: 16g_12_puntajes_pares_puntuar.parquet y 16g_12_candidatos_puntuar.csv")
    sys.exit(0)
if max(a.objetivos) <= 0:
    raise SystemExit("Se necesita al menos un objetivo mayor que 0")
rng_global = np.random.default_rng(a.semilla)

# ---------------------------------------------------------------- datos reales (igual que el script 16g_05)
ann = u.leer_anotaciones(a.anot)
ann["sesion"] = ann["Grupo"].str[:3]
ann["seg"] = (pd.to_timedelta(ann["time_start"]).dt.total_seconds() // 2).astype(int)
casos = pd.read_parquet(os.path.join(a.salida, "16g_02_casos.parquet")); casos = casos[casos["usable"]].copy()
ses = sorted(casos["sesion"].unique())
F = pd.read_parquet(os.path.join(a.salida, "16g_05_caract_personas.parquet")); F = F[F["sesion"].isin(ses)]
largo = []
for k in range(1, 5):
    d = ann[["sesion", "seg", f"speaker{k}"]].rename(columns={f"speaker{k}": "accion"}); d["track_id"] = k; largo.append(d)
Y = pd.concat(largo); Y["accion"] = Y["accion"].fillna("none").str.strip().str.lower()
D = F.merge(Y, on=["sesion", "seg", "track_id"], how="inner")
D["grupo"] = D["sesion"].str[:2]
cuenta = D["accion"].value_counts()
raras = set(cuenta[cuenta < MIN_CLASE].index) - {"none"}
D["clase"] = D["accion"].where(~D["accion"].isin(raras), "otra")
FEAT = [c for c in D.columns if c.endswith(("_m", "_sd", "_v")) or c.startswith("h") and c[1] in "LR" and c[2] == "_"]
grupos = sorted(D["grupo"].unique())

ruta08 = os.path.join(a.salida, "16g_08_por_accion.csv")
canales = {"cuerpo": ["cuerpo"], "cara": ["cara"], "ambos": ["cuerpo", "cara"]}[a.aumentar]
if os.path.exists(ruta08):
    t08 = pd.read_csv(ruta08)
    ACC = [x for x in t08.loc[t08["canal_donde_produce_mimetismo"].isin(canales), "accion"]]
else:
    ACC = (CUERPO_POR_DEFECTO if "cuerpo" in canales else []) + (CARA_POR_DEFECTO if "cara" in canales else [])
AUMENTAR = [c for c in ACC if c in set(D["clase"])]
p(f"Acciones que se aumentan (canal {a.aumentar}):", AUMENTAR)
p("Cantidades probadas (ejemplos por accion):", a.objetivos, "| maximo de variantes por ejemplo real:", a.max_por_ejemplo)

# ---------------------------------------------------------------- funciones de caracteristicas (copia fiel del 16g_05)
def feats_pose(S):
    cc = [c for c in S.columns if c.startswith("c_")]
    pts = [0] + list(range(11, 23))
    cols_pos = [f"b{i}_{e}" for i in pts for e in "xy"]
    g = S.groupby(["track_id", "seg"])
    m = g[cols_pos + cc].mean().add_suffix("_m"); sd = g[cols_pos + cc].std().add_suffix("_sd")
    v = S.sort_values(["track_id", "frame"]).assign(**{c: S.sort_values(["track_id", "frame"]).groupby("track_id")[c].diff().abs() for c in cols_pos})
    v = v.groupby(["track_id", "seg"])[cols_pos].mean().add_suffix("_v")
    return pd.concat([m, sd, v], axis=1).reset_index()

def normalizar_manos(H):
    """Coordenadas de cada mano relativas a la muneca y divididas por la distancia muneca-nudillo medio (igual que el 16g_05)."""
    base = H[["frame", "track_id", "hand"]].copy()
    mx = H["9_x"].values - H["0_x"].values; my = H["9_y"].values - H["0_y"].values
    esc = np.hypot(mx, my); esc = np.where(esc < 1e-6, np.nan, esc)
    for i in range(1, 21):
        base[f"{i}_x"] = (H[f"{i}_x"].values - H["0_x"].values) / esc
        base[f"{i}_y"] = (H[f"{i}_y"].values - H["0_y"].values) / esc
    return base

def feats_manos(base, S):
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

TIPS = [4, 8, 12, 16, 20]; PARES = [(4, 8), (8, 12), (12, 16), (16, 20)]
def feats_finas(base, S):
    """Caracteristicas finas de manos y dedos por (persona, segmento) y mano. Ver la descripcion al inicio."""
    cols = [f"{i}_{e}" for i in set(TIPS) | {9} for e in "xy"]
    b = base[["frame", "track_id", "hand"] + cols].merge(S[["frame", "track_id", "seg"]], on=["frame", "track_id"], how="inner")
    esc = np.hypot(b["9_x"].values, b["9_y"].values); esc = np.where(esc < 1e-6, np.nan, esc)
    for c in cols:
        b[c] = b[c].values / esc
    b = b.groupby(["track_id", "seg", "hand", "frame"], as_index=False)[cols].mean().sort_values(["track_id", "seg", "hand", "frame"])
    D_ = pd.DataFrame({"track_id": b["track_id"].values, "seg": b["seg"].values, "hand": b["hand"].values})
    for t in TIPS:
        D_[f"ext{t}"] = np.hypot(b[f"{t}_x"].values, b[f"{t}_y"].values)
    for (u1, u2) in PARES:
        D_[f"ap{u1}_{u2}"] = np.hypot(b[f"{u1}_x"].values - b[f"{u2}_x"].values, b[f"{u1}_y"].values - b[f"{u2}_y"].values)
    gk = ["track_id", "seg", "hand"]
    for t in TIPS:
        dx = b[f"{t}_x"].groupby([b["track_id"], b["seg"], b["hand"]]).diff(); dy = b[f"{t}_y"].groupby([b["track_id"], b["seg"], b["hand"]]).diff()
        D_[f"vel{t}"] = np.hypot(dx.values, dy.values)
    ag = {c: ["mean", "std"] for c in D_.columns if c.startswith(("ext", "ap"))}
    ag.update({c: ["mean"] for c in D_.columns if c.startswith("vel")})
    r = D_.groupby(gk).agg(ag); r.columns = [f"fn_{c}_{st}" for c, st in r.columns]
    r = r.reset_index(); res = None
    for mano in ("Left", "Right"):
        f = r[r["hand"] == mano].drop(columns="hand"); f.columns = [c if c in ("track_id", "seg") else c.replace("fn_", f"fn{mano[0]}_", 1) for c in f.columns]
        res = f if res is None else res.merge(f, on=["track_id", "seg"], how="outer")
    return res

COL_X = [f"{i}_x" for i in range(1, 21)]; COL_Y = [f"{i}_y" for i in range(1, 21)]
COLS_POSE_XY = [f"b{i}_{e}" for i in range(0, 23) for e in "xy"]

# ---------------------------------------------------------------- carga por sesion y comprobacion de equivalencia
def cargar_sesion(cod):
    S = pd.read_parquet(os.path.join(a.salida, f"16g_01_series_{cod}.parquet"))
    rutas = glob.glob(os.path.join(a.root, f"Grupo{cod[:2]}", cod[2], "*.handsTracker.parquet"))
    if not rutas:
        raise FileNotFoundError(f"falta handsTracker de {cod}")
    return S, normalizar_manos(pd.read_parquet(rutas[0]))

p("\nComprobacion: recalcular las caracteristicas reales de una sesion con las funciones de este script y compararlas con el 16g_05")
cod0 = ses[0]; S0, B0 = cargar_sesion(cod0)
f0 = feats_pose(S0).merge(feats_manos(B0, S0), on=["track_id", "seg"], how="left")
r0 = F[F["sesion"] == cod0].merge(f0, on=["track_id", "seg"], suffixes=("", "_nuevo"))
dif = max(float(np.nanmax(np.abs(r0[c].values - r0[c + "_nuevo"].values))) for c in FEAT if c + "_nuevo" in r0.columns)
p(f"  sesion {cod0}: diferencia maxima = {dif:.2e}")
if dif > 1e-6:
    raise SystemExit("Las caracteristicas recalculadas no coinciden con las del 16g_05: no se genera nada.")

# ---------------------------------------------------------------- escalas de ruido ancladas en los datos reales
dp = []; dh = []; FINr = []
for cod in ses:
    S, B = cargar_sesion(cod) if cod != cod0 else (S0, B0)
    if a.rep == "fina":
        FINr.append(feats_finas(B, S).assign(sesion=cod))
    S = S.sort_values(["track_id", "frame"])
    dp.append(S.groupby("track_id")[[f"b{i}_{e}" for i in range(11, 23) for e in "xy"]].diff().abs().stack().median())
    Bm = B[B["hand"] == "Right"].sort_values(["track_id", "frame"])
    dh.append(Bm.groupby("track_id")[COL_X + COL_Y].diff().abs().stack().median())
SIG_POSE = RUIDO * float(np.nanmedian(dp)); SIG_MANO = RUIDO * float(np.nanmedian(dh))
if a.rep == "fina":
    FINr = pd.concat(FINr, ignore_index=True)
    FEAT_FINAS = [c for c in FINr.columns if c.startswith("fn")]
    D = D.merge(FINr, on=["sesion", "track_id", "seg"], how="left")
    FEAT = FEAT + FEAT_FINAS
    p(f"Caracteristicas finas agregadas: {len(FEAT_FINAS)} (total {len(FEAT)}); con dato en {D[FEAT_FINAS[0]].notna().mean():.1%} de los segmentos-persona")
p(f"\nRuido de los sinteticos (la mitad de la mediana del cambio entre cuadros reales): pose = {SIG_POSE:.5f} | manos = {SIG_MANO:.5f}")

# ---------------------------------------------------------------- generacion de sinteticos
def rot(x, y, ang):
    c, s = np.cos(ang), np.sin(ang)
    return c * x - s * y, s * x + c * y

def variante(Pm, hand_por_cuadro, n, rng):
    """Pm: matriz (n, k) de pose y descriptores; hand_por_cuadro: lista de matrices (filas de manos) por cuadro original."""
    w = rng.uniform(VEL_MIN, VEL_MAX)
    idx = np.clip(np.rint((np.arange(n) - n / 2) * w + n / 2).astype(int), 0, n - 1)
    P = Pm[idx].copy()
    th = np.deg2rad(rng.uniform(-ROT_POSE, ROT_POSE)); sc = rng.uniform(ESC_MIN, ESC_MAX)
    X = P[:, :len(COLS_POSE_XY):2]; Yv = P[:, 1:len(COLS_POSE_XY):2]
    X2, Y2 = rot(X, Yv, th); X2 = X2 * sc + rng.normal(0, SIG_POSE, X.shape); Y2 = Y2 * sc + rng.normal(0, SIG_POSE, Yv.shape)
    P[:, :len(COLS_POSE_XY):2] = X2; P[:, 1:len(COLS_POSE_XY):2] = Y2
    th_m = {"Left": np.deg2rad(rng.uniform(-ROT_MANO, ROT_MANO)), "Right": np.deg2rad(rng.uniform(-ROT_MANO, ROT_MANO))}
    sc_m = {"Left": rng.uniform(ESC_MIN, ESC_MAX), "Right": rng.uniform(ESC_MIN, ESC_MAX)}
    filas = []
    for k in range(n):
        h = hand_por_cuadro[idx[k]]
        if h is None or len(h) == 0:
            continue
        h = h.copy()
        for j, mano in enumerate(("Left", "Right")):
            sel = h[:, 0] == j
            if sel.any():
                hx, hy = rot(h[sel, 1:21], h[sel, 21:41], th_m[mano])
                h[sel, 1:21] = hx * sc_m[mano] + rng.normal(0, SIG_MANO, hx.shape)
                h[sel, 21:41] = hy * sc_m[mano] + rng.normal(0, SIG_MANO, hy.shape)
        filas.append(np.column_stack([np.full(len(h), k + 1.0), h]))
    return P, (np.vstack(filas) if filas else np.empty((0, 42)))

def generar_pool(objetivo_max):
    """Genera, por sesion, variantes de los ejemplos reales de las acciones a aumentar. Devuelve DataFrame con caracteristicas."""
    n_clase = D["clase"].value_counts()
    V = {c: int(min(a.max_por_ejemplo, max(1, np.ceil(objetivo_max / n_clase[c]) - 1))) for c in AUMENTAR if n_clase[c] < objetivo_max}
    p("\nVariantes sinteticas por ejemplo real y accion (para la mayor cantidad probada):", V)
    total = sum(int((D["clase"] == c).sum()) * v for c, v in V.items())
    prog = u.Progreso(len(ses), f"Generando sinteticos (hasta {total} segmentos); una unidad = una sesion", log)
    partes = []; sid = 1
    for k, cod in enumerate(ses, 1):
        prog.actualizar(f"sesion {cod} ({k}/{len(ses)})")
        S, B = cargar_sesion(cod) if cod != cod0 else (S0, B0)
        S = S.sort_values(["track_id", "frame"]).reset_index(drop=True)
        cols_s = COLS_POSE_XY + [c for c in S.columns if c.startswith("c_")]
        ej = D[(D["sesion"] == cod) & (D["clase"].isin(V))][["track_id", "seg", "clase"]]
        if len(ej) == 0:
            prog.actualizar(avance=1); continue
        # filas de manos por (persona, cuadro), como matrices (indicador de mano, 20 x, 20 y)
        B = B.copy(); B["j"] = (B["hand"] == "Right").astype(int)
        Bm = B[["track_id", "frame", "j"] + COL_X + COL_Y].to_numpy()
        orden = np.lexsort((Bm[:, 1], Bm[:, 0])); Bm = Bm[orden]
        claves = Bm[:, 0].astype(np.int64) * 10_000_000 + Bm[:, 1].astype(np.int64)
        gS = {kk: g for kk, g in S.groupby(["track_id", "seg"])}
        rng = np.random.default_rng(a.semilla + k)
        SP, SF, SM = [], [], []
        for _, r in ej.iterrows():
            g = gS.get((r["track_id"], r["seg"]))
            if g is None or len(g) < 5:
                continue
            Pm = g[cols_s].to_numpy(); frames = g["frame"].to_numpy(); n = len(g)
            hpc = []
            for fr in frames:
                kk = int(r["track_id"]) * 10_000_000 + int(fr)
                lo, hi = np.searchsorted(claves, kk, "left"), np.searchsorted(claves, kk, "right")
                hpc.append(Bm[lo:hi, 2:] if hi > lo else None)
            for _ in range(V[r["clase"]]):
                P, Hh = variante(Pm, hpc, n, rng)
                SP.append(pd.DataFrame(P, columns=cols_s).assign(frame=np.arange(1, n + 1), track_id=sid, seg=0, clase=r["clase"], grupo=cod[:2], sesion=cod))
                if len(Hh):
                    SM.append(pd.DataFrame(Hh, columns=["frame", "j"] + COL_X + COL_Y).assign(track_id=sid, hand=lambda d: np.where(d["j"] == 0, "Left", "Right")).drop(columns="j"))
                sid += 1
        if SP:
            SS = pd.concat(SP, ignore_index=True)
            f = feats_pose(SS[["frame", "track_id", "seg"] + cols_s]).merge(
                feats_manos(pd.concat(SM, ignore_index=True)[["frame", "track_id", "hand"] + COL_X + COL_Y], SS[["frame", "track_id", "seg"]]),
                on=["track_id", "seg"], how="left")
            if a.rep == "fina":
                bm = pd.concat(SM, ignore_index=True)[["frame", "track_id", "hand"] + COL_X + COL_Y]
                f = f.merge(feats_finas(bm, SS[["frame", "track_id", "seg"]]), on=["track_id", "seg"], how="left")
            meta = SS.groupby("track_id")[["clase", "grupo", "sesion"]].first().reset_index()
            partes.append(f.merge(meta, on="track_id"))
        prog.actualizar(avance=1)
    prog.cerrar()
    return pd.concat(partes, ignore_index=True)

POOL = generar_pool(max(max(a.objetivos), a.reserva))
for c in FEAT:
    if c not in POOL.columns:
        POOL[c] = np.nan
p("Sinteticos generados en total:", len(POOL)); p(POOL["clase"].value_counts().to_string())
pd.DataFrame({"clase": POOL["clase"].value_counts().index, "generados": POOL["clase"].value_counts().values,
              "reales": [int((D["clase"] == c).sum()) for c in POOL["clase"].value_counts().index]}).to_csv(os.path.join(a.salida, PRE + "sinteticos.csv"), index=False)

# fidelidad: los sinteticos se parecen a los reales de su misma accion? (media y dispersion de cada caracteristica)
filas = []
for c in sorted(POOL["clase"].unique()):
    r = D.loc[D["clase"] == c, FEAT]; sy = POOL.loc[POOL["clase"] == c, FEAT]
    sd_r = r.std().replace(0, np.nan)
    filas.append(dict(clase=c, reales=len(r), sinteticos=len(sy),
                      dif_media_en_desv_reales=float(((sy.mean() - r.mean()).abs() / sd_r).median()),
                      razon_desviaciones=float((sy.std() / sd_r).median())))
FID = pd.DataFrame(filas); FID.to_csv(os.path.join(a.salida, PRE + "fidelidad.csv"), index=False)
p("\nFIDELIDAD (mediana sobre las caracteristicas; dif. de medias en desviaciones de los reales: cerca de 0 es parecido; razon de desviaciones: cerca de 1 es parecido):")
p(FID.round(3).to_string(index=False))

# ---------------------------------------------------------------- evaluacion: un grupo fuera, con sinteticos solo en el entrenamiento
# ---------------------------------------------------------------- entrenamiento del modelo final (16 grupos + sinteticos)
obj = a.objetivos[0]
rng = np.random.default_rng(a.semilla)
extra = []
for c in AUMENTAR:
    falta = obj - int((D["clase"] == c).sum())
    pool_c = POOL[POOL["clase"] == c]
    if falta > 0 and len(pool_c):
        extra.append(pool_c.sample(n=min(falta, len(pool_c)), random_state=int(rng.integers(1 << 30))))
sint = pd.concat(extra, ignore_index=True) if extra else POOL.iloc[0:0]
entr = pd.concat([D[FEAT + ["clase"]], sint[FEAT + ["clase"]]], ignore_index=True)
p(f"\nEntrenamiento final: {len(D)} segmentos-persona reales de {D['grupo'].nunique()} grupos + {len(sint)} sinteticos ({len(sint) / len(entr):.1%} del entrenamiento)")
modelo = make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestClassifier(n_estimators=a.arboles, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=a.semilla))
modelo.fit(entr[FEAT], entr["clase"])
conf = dict(objetivo=obj, max_por_ejemplo=a.max_por_ejemplo, arboles=a.arboles, semilla=a.semilla, representacion=a.rep, grupos=grupos,
            n_reales=int(len(D)), n_sinteticos=int(len(sint)), acciones_aumentadas=AUMENTAR, fecha=datetime.date.today().isoformat())
M = dict(modelo=modelo, caracteristicas=FEAT, configuracion=conf)
joblib.dump(M, os.path.join(a.salida, PRE + "modelo_final.joblib"))
p("Modelo guardado:", os.path.join(a.salida, PRE + "modelo_final.joblib"))

# puntaje de los pares de los 16 grupos (EN MUESTRA: el modelo ya vio estos grupos; no es una evaluacion)
cs = casos.copy()
Fall = D[["sesion", "seg", "track_id"] + FEAT].copy()
ct = puntuar_pares(M, Fall, cs)
ct[["sesion", "grupo", "seg", "pareja", "canal", "y", "puntaje"]].to_parquet(os.path.join(a.salida, PRE + "puntajes_pares.parquet"), index=False)
candidatos(ct).to_csv(os.path.join(a.salida, PRE + "candidatos.csv"), index=False)
d = ct.dropna(subset=["puntaje"])
en_muestra = {c: float(average_precision_score(d.loc[d["canal"] == c, "y"], d.loc[d["canal"] == c, "puntaje"]) / d.loc[d["canal"] == c, "y"].mean()) for c in ("cuerpo", "cara")}

# ficha del modelo, con el desempeno VALIDADO (dejando un grupo fuera) leido del 16g_10
valid = ""
for nombre in ("16g_10_base_resultados.csv", "16g_10_base_s1_resultados.csv"):
    ruta = os.path.join(a.salida, nombre)
    if os.path.exists(ruta):
        v = pd.read_csv(ruta); v = v[v["objetivo"] == obj]
        if len(v):
            valid += f"\n  {nombre} (objetivo {obj}):\n" + v[["canal", "pr_auc", "ic_inf", "ic_sup", "x_azar", "cumple_global", "grupos_que_cumplen"]].round(3).to_string(index=False)
p("\n" + "=" * 100); p("FICHA DEL MODELO FINAL"); p("=" * 100)
p("Configuracion:", json.dumps(conf, ensure_ascii=False))
p("Entrenado con: etapa 1 Random Forest (23 clases de accion maximo; las de menos de 40 ejemplos se agrupan en 'otra'); etapa 2 sin entrenamiento: puntaje del par = suma de p_i(c) * p_j(c) sin none ni otra.")
p("Desempeno VALIDADO (un grupo fuera, 16 pruebas; es el que se atribuye al modelo):" + (valid if valid else " (no se encontraron los resultados del 16g_10 en la carpeta de salidas)"))
p("\nPuntaje en muestra (el modelo ya vio estos grupos; NO es una evaluacion y sobreestima el desempeno): veces el azar cuerpo = %.2f, cara = %.2f" % (en_muestra["cuerpo"], en_muestra["cara"]))
p("Uso: python 16g_12_modelo_final.py --modo puntuar --grupos 01 15")
log.guardar(os.path.join(a.salida, PRE + "ficha_modelo.txt"))
