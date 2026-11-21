"""
16g_08_diagnostico_etapa1.py   (diagnostico de la etapa 1 del enfoque en dos etapas)
El script 16g_07 mostro que, con la accion REAL de cada persona, la regla "misma accion" llega a 12x (cuerpo) y 14x (cara) sobre
el azar, pero el enfoque con la accion RECONOCIDA llega a 1,7x (cuerpo) y 6,8x (cara). La diferencia esta en la etapa 1.
Este script mide, con el mismo Random Forest y la misma validacion (un grupo fuera), cuan bien se reconoce cada accion:
exhaustividad, precision y F1 por accion, agrupadas segun el canal en que esa accion produce mimetismo anotado (segun 16g_07_por_accion.csv),
y las confusiones mas frecuentes. No entrena nada nuevo para el informe: es solo para entender donde falla la etapa 1.

Uso (desde la carpeta scripts16g):   python 16g_08_diagnostico_etapa1.py       (necesita haber corrido 16g_05 y 16g_07)
Entradas: Root/mimicry_indicators.xlsx, salidas_16g/16g_05_caract_personas.parquet, salidas_16g/16g_07_por_accion.csv
Salidas:  salidas_16g/16g_08_diagnostico_etapa1.txt, 16g_08_por_accion.csv
"""
import os, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
import util16g as u

warnings.filterwarnings("ignore")
MIN_CLASE = 40; SEMILLA = 0
ap = u.parser("Diagnostico etapa 1")
ap.add_argument("--arboles", type=int, default=300)
a = ap.parse_args()
log = u.Log(); p = log.p

ann = u.leer_anotaciones(a.anot)
ann["sesion"] = ann["Grupo"].str[:3]
ann["seg"] = (pd.to_timedelta(ann["time_start"]).dt.total_seconds() // 2).astype(int)
F = pd.read_parquet(os.path.join(a.salida, "16g_05_caract_personas.parquet"))
if a.grupos:
    F = F[F["sesion"].str[:2].isin([g.zfill(2) for g in a.grupos])]
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

prog = u.Progreso(len(grupos), "Etapa 1 con un grupo fuera (una unidad = un grupo)", log)
res = []
for g in grupos:
    prog.actualizar(f"grupo {g}: entrenando con los demas")
    tr, te = D[D["grupo"] != g], D[D["grupo"] == g]
    m = make_pipeline(SimpleImputer(strategy="median"),
                      RandomForestClassifier(n_estimators=a.arboles, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=SEMILLA))
    m.fit(tr[FEAT], tr["clase"])
    res.append(pd.DataFrame({"grupo": g, "real": te["clase"].values, "pred": m.predict(te[FEAT])}))
    prog.actualizar(avance=1)
prog.cerrar()
R = pd.concat(res, ignore_index=True)

# tipo de accion segun donde produce mimetismo anotado (16g_07)
tipo = {}
ruta7 = os.path.join(a.salida, "16g_07_por_accion.csv")
if os.path.exists(ruta7):
    T = pd.read_csv(ruta7)
    for ac, d in T.groupby("accion"):
        b = d[d["canal"] == "cuerpo"]["%_con_mimetismo"].max() if (d["canal"] == "cuerpo").any() else 0
        c = d[d["canal"] == "cara"]["%_con_mimetismo"].max() if (d["canal"] == "cara").any() else 0
        tipo[ac] = "cuerpo" if b >= 50 and c < 50 else ("cara" if c >= 50 and b < 50 else "ninguno/mixto")
clases = sorted(R["real"].unique())
pr, rc, f1, sp = precision_recall_fscore_support(R["real"], R["pred"], labels=clases, zero_division=0)
T8 = pd.DataFrame({"accion": clases, "soporte": sp, "exhaustividad": rc, "precision": pr, "f1": f1})
T8["canal_donde_produce_mimetismo"] = T8["accion"].map(tipo).fillna("sin dato")
T8 = T8.sort_values(["canal_donde_produce_mimetismo", "soporte"], ascending=[True, False])
T8.to_csv(os.path.join(a.salida, "16g_08_por_accion.csv"), index=False)
pd.set_option("display.width", 200)
p(f"Segmentos-persona evaluados (fuera de muestra, un grupo fuera): {len(R)} | exactitud global = {(R['real'] == R['pred']).mean():.3f}")
p("\nPOR ACCION (exhaustividad = de las veces que la accion ocurre, cuantas reconoce; precision = de las veces que la predice, cuantas acierta):")
p(T8.round(3).to_string(index=False))
p("\nPOR TIPO DE ACCION (ponderado por soporte; excluye 'none'):")
x = T8[T8["accion"] != "none"]
for t, d in x.groupby("canal_donde_produce_mimetismo"):
    w = d["soporte"].values
    p(f"  {t:15s} acciones={len(d):2d} soporte={int(w.sum()):6d} exhaustividad={np.average(d['exhaustividad'], weights=w):.3f} "
      f"precision={np.average(d['precision'], weights=w):.3f} f1={np.average(d['f1'], weights=w):.3f}")
C = pd.crosstab(R["real"], R["pred"]); off = [(i, j, int(C.loc[i, j])) for i in C.index for j in C.columns if i != j and C.loc[i, j] > 0]
off.sort(key=lambda t: -t[2])
p("\nCONFUSIONES MAS FRECUENTES (real -> predicha, veces):")
for i, j, n in off[:15]:
    p(f"  {i} -> {j}: {n}")
log.guardar(os.path.join(a.salida, "16g_08_diagnostico_etapa1.txt"))
