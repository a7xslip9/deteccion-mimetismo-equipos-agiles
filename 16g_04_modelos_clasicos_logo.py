"""
16g_04_modelos_clasicos_logo.py   (MS3 + MS4; equivale al script 06, ahora con los 16 grupos)
Compara, por canal (cuerpo, cara), lineas base y clasificadores sobre las medidas de sincronia.

Validacion principal: DEJAR UN GRUPO FUERA ("leave-one-group-out"). Se entrena con 15 grupos y se prueba con el
grupo que falta; se repite 16 veces, una por grupo. Ningun caso de prueba comparte grupo (ni personas) con el entrenamiento.
Validacion secundaria (por sesion): entrena con todas las sesiones A y prueba con las B, y viceversa (igual que en la prueba de 2 grupos).
Dentro del entrenamiento, el umbral de decision se elige con predicciones fuera de muestra (validacion cruzada por grupo, 3 partes).

Lineas base: azar y cada medida de sincronia sola.  Modelos: regresion logistica, Random Forest, gradient boosting.
Metrica principal: PR-AUC (average precision) contra la prevalencia. IC 95 % por bootstrap en bloques de 10 segmentos.

Uso (desde la carpeta scripts16g):   python 16g_04_modelos_clasicos_logo.py
Entradas: salidas_16g/16g_03_caracteristicas.parquet
Salidas:  salidas_16g/16g_04_resultados.csv (una fila por prueba/modelo), 16g_04_predicciones.parquet, 16g_04_modelos_clasicos_logo.txt
Requiere: pandas, pyarrow, numpy, scikit-learn
"""
import os, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import util16g as u

warnings.filterwarnings("ignore")
SEMILLA = 0; N_BOOT = 500; BLOQUE = 10; PARTES_INTERNAS = 3
a = u.parser("MS3/MS4 clasicos, un grupo fuera", root=False, anot=False).parse_args()
log = u.Log(); p = log.p

R = pd.read_parquet(os.path.join(a.salida, "16g_03_caracteristicas.parquet"))
if a.grupos:
    R = R[R["grupo"].isin([g.zfill(2) for g in a.grupos])]
SINC = ["xc_pos_max", "xc_pos_lag", "dtw_pos", "xc_vel_max", "xc_vel_lag", "dtw_vel", "dur_sim"]
TODAS = SINC + ["mov_min", "mov_max"]
SENTIDO = {"xc_pos_max": 1, "xc_vel_max": 1, "dur_sim": 1, "dtw_pos": -1, "dtw_vel": -1, "xc_pos_lag": -1, "xc_vel_lag": -1}
try:
    gb = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEMILLA)
except TypeError:
    gb = HistGradientBoostingClassifier(random_state=SEMILLA)
MODELOS = {
    "regresion_logistica": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
    "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=SEMILLA),
    "gradient_boosting": gb,
}

def umbral_f1(y, s):
    pr, rc, th = precision_recall_curve(y, s)
    f = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return th[int(np.argmax(f))] if len(th) else 0.5

def ic_boot(y, s, bloque_id):
    rng = np.random.default_rng(SEMILLA); ids = np.unique(bloque_id); vals = []
    por = {b: np.where(bloque_id == b)[0] for b in ids}
    for _ in range(N_BOOT):
        sel = np.concatenate([por[b] for b in rng.choice(ids, len(ids))])
        if y[sel].sum() == 0: continue
        vals.append(average_precision_score(y[sel], s[sel]))
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)

filas = []; preds = []
def registrar(esq, canal, prueba, modelo, te, s, pred=None):
    y = te["y"].values; lo, hi = ic_boot(y, s, te["bloque"].values)
    r = dict(esquema=esq, canal=canal, prueba=prueba, modelo=modelo, n=len(y), positivos=int(y.sum()), prevalencia=y.mean(),
             pr_auc=average_precision_score(y, s), ic_inf=lo, ic_sup=hi,
             roc_auc=roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan)
    if pred is not None:
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        r["precision"] = tp / max(tp + fp, 1); r["exhaustividad"] = tp / max(tp + fn, 1); r["f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    filas.append(r)
    preds.append(pd.DataFrame({"esquema": esq, "canal": canal, "prueba": prueba, "modelo": modelo, "sesion": te["sesion"].values,
                               "grupo": te["grupo"].values, "seg": te["seg"].values, "pareja": te["pareja"].values,
                               "y": y, "score": s, "bloque": te["bloque"].values}))

R["bloque"] = R["sesion"] + "_" + (R["seg"] // BLOQUE).astype(str)
R["letra"] = R["sesion"].str[-1]
grupos = sorted(R["grupo"].unique())
PRUEBAS = [("logo", g, f"grupo={g}", lambda D, g=g: (D[D["grupo"] != g], D[D["grupo"] == g])) for g in grupos]
PRUEBAS += [("por_sesion", "B", "entrena A -> prueba B", lambda D: (D[D["letra"] == "A"], D[D["letra"] == "B"])),
            ("por_sesion", "A", "entrena B -> prueba A", lambda D: (D[D["letra"] == "B"], D[D["letra"] == "A"]))]
p(f"Grupos: {len(grupos)} ({' '.join(grupos)}) | casos: {len(R)} | pruebas por canal: {len(PRUEBAS)} (un grupo fuera + por sesion)")

prog = u.Progreso(2 * len(PRUEBAS), "MS3/MS4: modelos clasicos (una unidad = una prueba en un canal; incluye 3 modelos)", log)
for canal in ("cuerpo", "cara"):
    D = R[R["canal"] == canal].reset_index(drop=True)
    for esq, _, nombre, corte in PRUEBAS:
        prog.actualizar(f"{canal} | {nombre}")
        tr, te = corte(D)
        if te["y"].sum() == 0 or tr["y"].sum() == 0:
            p(f"  [{canal}] {nombre}: sin positivos en prueba o entrenamiento, se omite"); prog.actualizar(avance=1); continue
        rng = np.random.default_rng(SEMILLA)
        registrar(esq, canal, nombre, "base_azar", te, rng.random(len(te)))
        for f in SINC:
            registrar(esq, canal, nombre, f"base_{f}", te, SENTIDO[f] * te[f].values)
        # grupos internos para el umbral: por grupo si hay suficientes, si no por sesion
        gcol = "grupo" if tr["grupo"].nunique() >= 3 else "sesion"
        gr = tr[gcol].values; n_pl = min(PARTES_INTERNAS, len(np.unique(gr)))
        for nom, mod in MODELOS.items():
            prog.actualizar(f"{canal} | {nombre} | {nom}")
            oof = cross_val_predict(clone(mod), tr[TODAS], tr["y"], groups=gr, cv=GroupKFold(n_splits=n_pl), method="predict_proba")[:, 1]
            th = umbral_f1(tr["y"].values, oof)
            m = clone(mod).fit(tr[TODAS], tr["y"])
            s = m.predict_proba(te[TODAS])[:, 1]
            registrar(esq, canal, nombre, nom, te, s, pred=(s >= th).astype(int))
        prog.actualizar(avance=1)
prog.cerrar()

Res = pd.DataFrame(filas)
Res.to_csv(os.path.join(a.salida, "16g_04_resultados.csv"), index=False)
pd.concat(preds, ignore_index=True).to_parquet(os.path.join(a.salida, "16g_04_predicciones.parquet"), index=False)
pd.set_option("display.width", 250)
M = Res[Res["modelo"].isin(MODELOS)]
for esq in M["esquema"].unique():
    for canal in ("cuerpo", "cara"):
        d = M[(M["esquema"] == esq) & (M["canal"] == canal)].copy()
        d["x_azar"] = d["pr_auc"] / d["prevalencia"]
        p("=" * 100); p(f"ESQUEMA: {esq} | CANAL: {canal}  (x_azar = PR-AUC / prevalencia; la meta es >= 2)"); p("=" * 100)
        p(d.pivot_table(index="prueba", columns="modelo", values="x_azar").round(2).to_string())
        p("\nPromedio de x_azar por modelo:", d.groupby("modelo")["x_azar"].mean().round(2).to_dict())
log.guardar(os.path.join(a.salida, "16g_04_modelos_clasicos_logo.txt"))
