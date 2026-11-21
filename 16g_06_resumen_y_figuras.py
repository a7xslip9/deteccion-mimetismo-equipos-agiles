"""
16g_06_resumen_y_figuras.py
Junta los resultados de 16g_04 (medidas clasicas) y 16g_05 (dos etapas) y los evalua con el criterio fijado de antemano.

CRITERIO (el mismo de la prueba de 2 grupos, extendido a los 16 grupos): un enfoque "supera al azar" en un canal si
  PR-AUC >= 2 x prevalencia   Y   el limite inferior del IC 95 % queda sobre la prevalencia.
Se informa de dos formas:
  (a) GLOBAL: se juntan los puntajes de los 16 grupos de prueba (cada uno predicho por un modelo que no lo vio) y se calcula
      una sola PR-AUC, con IC por bootstrap en bloques de 10 segmentos.
  (b) POR GRUPO: en cuantos de los 16 grupos de prueba se cumple el criterio, y el promedio y la mediana de PR-AUC / prevalencia.
El criterio se aplica igual a todos los enfoques; no se elige el mejor y se descarta el resto.

Uso (desde la carpeta scripts16g):   python 16g_06_resumen_y_figuras.py
Entradas: salidas_16g/16g_02_casos.parquet, 16g_04_resultados.csv, 16g_04_predicciones.parquet,
          16g_05_resultados.csv, 16g_05_puntajes_pareja.parquet
Salidas:  salidas_16g/16g_06_resumen.txt, 16g_06_resumen_global.csv, 16g_06_resumen_por_grupo.csv,
          salidas_16g/figuras/fig_16g_resultados.png, fig_16g_casos.png
Requiere: pandas, pyarrow, numpy, scikit-learn, matplotlib
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import util16g as u

N_BOOT = 500; SEMILLA = 0; META = 2.0
a = u.parser("Resumen y figuras (16 grupos)", root=False, anot=False).parse_args()
log = u.Log(); p = log.p
os.makedirs(os.path.join(a.salida, "figuras"), exist_ok=True)
S = a.salida
NOMBRES = {"regresion_logistica": "Medidas clásicas + reg. logística", "random_forest": "Medidas clásicas + Random Forest",
           "gradient_boosting": "Medidas clásicas + gradient boosting", "dos_etapas": "Dos etapas (acción → pareja), pose + manos",
           "dos_etapas_pose": "Dos etapas, pose sola"}
ORDEN = ["regresion_logistica", "random_forest", "gradient_boosting", "dos_etapas", "dos_etapas_pose"]

# ---------- datos: un grupo fuera ----------
R4 = pd.read_csv(os.path.join(S, "16g_04_resultados.csv"))
R5 = pd.read_csv(os.path.join(S, "16g_05_resultados.csv"))
R5["modelo"] = np.where(R5["caract"] == "pose + manos", "dos_etapas", "dos_etapas_pose")
P4 = pd.read_parquet(os.path.join(S, "16g_04_predicciones.parquet"))
P5 = pd.read_parquet(os.path.join(S, "16g_05_puntajes_pareja.parquet"))
P5 = P5.rename(columns={"puntaje": "score"})
P5["modelo"] = np.where(P5["caract"] == "pose + manos", "dos_etapas", "dos_etapas_pose")
P5 = P5[["esquema", "canal", "prueba", "modelo", "sesion", "grupo", "seg", "pareja", "y", "score", "bloque"]].dropna(subset=["score"])
PRED = pd.concat([P4[P4["modelo"].isin(ORDEN)], P5], ignore_index=True)
RES = pd.concat([R4[R4["modelo"].isin(ORDEN)], R5], ignore_index=True)
RES["x_azar"] = RES["pr_auc"] / RES["prevalencia"]
RES["cumple"] = (RES["pr_auc"] >= META * RES["prevalencia"]) & (RES["ic_inf"] > RES["prevalencia"])

def ic_boot(y, s, blq):
    rng = np.random.default_rng(SEMILLA); ids = np.unique(blq); por = {b: np.where(blq == b)[0] for b in ids}; v = []
    for _ in range(N_BOOT):
        sel = np.concatenate([por[b] for b in rng.choice(ids, len(ids))])
        if y[sel].sum(): v.append(average_precision_score(y[sel], s[sel]))
    return np.percentile(v, [2.5, 97.5])

def global_y_por_grupo(esq, etiqueta):
    glob_ = []; filas = []
    for canal in ("cuerpo", "cara"):
        for m in ORDEN:
            d = PRED[(PRED["esquema"] == esq) & (PRED["canal"] == canal) & (PRED["modelo"] == m)]
            if d.empty or d["y"].sum() == 0: continue
            y, s, b = d["y"].values, d["score"].values, d["bloque"].values
            lo, hi = ic_boot(y, s, b); prev = y.mean(); pa = average_precision_score(y, s)
            r = RES[(RES["esquema"] == esq) & (RES["canal"] == canal) & (RES["modelo"] == m)]
            glob_.append(dict(validacion=etiqueta, canal=canal, enfoque=m, pruebas=len(r), casos=len(y), positivos=int(y.sum()),
                              prevalencia=prev, pr_auc=pa, ic_inf=lo, ic_sup=hi, x_azar=pa / prev,
                              cumple_global=bool(pa >= META * prev and lo > prev), pruebas_que_cumplen=int(r["cumple"].sum()),
                              x_azar_promedio=r["x_azar"].mean(), x_azar_mediana=r["x_azar"].median()))
    return pd.DataFrame(glob_)

G = pd.concat([global_y_por_grupo("logo", "un grupo fuera"), global_y_por_grupo("por_sesion", "por sesion (A<->B)")], ignore_index=True)
G.to_csv(os.path.join(S, "16g_06_resumen_global.csv"), index=False)
RES.to_csv(os.path.join(S, "16g_06_resumen_por_grupo.csv"), index=False)

pd.set_option("display.width", 250)
n_g = RES[RES["esquema"] == "logo"]["prueba"].nunique()
p("CRITERIO: PR-AUC >= 2 x prevalencia y limite inferior del IC 95 % sobre la prevalencia.")
for val in ("un grupo fuera", "por sesion (A<->B)"):
    p("\n" + "=" * 100); p(f"VALIDACION: {val}"); p("=" * 100)
    for canal in ("cuerpo", "cara"):
        d = G[(G["validacion"] == val) & (G["canal"] == canal)].copy()
        d["enfoque"] = d["enfoque"].map(NOMBRES)
        d["IC95%"] = d.apply(lambda r: f"{r.ic_inf:.3f}-{r.ic_sup:.3f}", axis=1)
        d["cumple"] = d["cumple_global"].map({True: "SI", False: "no"})
        p(f"\n[{canal}]  GLOBAL (todas las pruebas juntas) y por prueba")
        p(d[["enfoque", "positivos", "prevalencia", "pr_auc", "IC95%", "x_azar", "cumple", "pruebas_que_cumplen", "pruebas",
             "x_azar_promedio", "x_azar_mediana"]].round(3).to_string(index=False))
p("\nLectura: x_azar = PR-AUC / prevalencia (1 = azar, 2 = meta). 'pruebas_que_cumplen' cuenta en cuantas de las pruebas "
  f"(en 'un grupo fuera' son {n_g}) el criterio se cumple por separado.")
# veredicto neutro por canal
p("\n" + "=" * 100); p("RESUMEN DEL CRITERIO (validacion un grupo fuera)"); p("=" * 100)
for canal in ("cuerpo", "cara"):
    d = G[(G["validacion"] == "un grupo fuera") & (G["canal"] == canal)]
    ok = d[d["cumple_global"]]["enfoque"].map(NOMBRES).tolist()
    p(f"[{canal}] enfoques que cumplen el criterio global: {ok if ok else 'ninguno'}")
    p(f"         mejor x_azar global: {d['x_azar'].max():.2f} ({NOMBRES[d.loc[d['x_azar'].idxmax(), 'enfoque']]})")

# ---------- figuras ----------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COL = {"regresion_logistica": "#9aa5b1", "random_forest": "#6b7a8f", "gradient_boosting": "#3d5a80", "dos_etapas": "#2a9d8f", "dos_etapas_pose": "#8ac6bd"}
ETIQ = {"regresion_logistica": "Reg.\nlogística", "random_forest": "Random\nForest", "gradient_boosting": "Gradient\nboosting",
        "dos_etapas": "Dos etapas\npose+manos", "dos_etapas_pose": "Dos etapas\npose sola"}
ORD = [m for m in ORDEN if m in set(RES["modelo"])]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
rng = np.random.default_rng(1)
for ax, canal in zip(axs, ("cuerpo", "cara")):
    for k, m in enumerate(ORD):
        d = RES[(RES["esquema"] == "logo") & (RES["canal"] == canal) & (RES["modelo"] == m)]
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(d)), d["x_azar"], s=26, color=COL[m], alpha=0.75, zorder=3)
        g = G[(G["validacion"] == "un grupo fuera") & (G["canal"] == canal) & (G["enfoque"] == m)]
        if len(g):
            g = g.iloc[0]; xa = g["x_azar"]; lo_ = g["ic_inf"] / g["prevalencia"]; hi_ = g["ic_sup"] / g["prevalencia"]
            ax.errorbar([k], [xa], yerr=[[xa - lo_], [hi_ - xa]], fmt="D", color="black", ms=7, capsize=3, zorder=4,
                        label="Global (IC 95 %)" if (k == 0 and canal == "cuerpo") else None)
    ax.axhline(1, color="#c1121f", ls="--", lw=1.3, label="Azar (1×)" if canal == "cuerpo" else None)
    ax.axhline(META, color="#2a6f97", ls=":", lw=1.3, label="Meta (2×)" if canal == "cuerpo" else None)
    ax.set_xticks(range(len(ORD))); ax.set_xticklabels([ETIQ[m] for m in ORD], fontsize=8)
    ax.set_title(f"Canal {canal}"); ax.set_xlim(-0.6, len(ORD) - 0.4)
for ax in axs:
    ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50, 100]); ax.set_yticklabels(["0,5", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_ylim(0.5, 150); ax.grid(axis="y", color="#e5e5e5", lw=0.6, zorder=0)
axs[0].set_ylabel("PR-AUC / prevalencia  (veces sobre el azar, escala log.)")
axs[0].scatter([], [], s=26, color="#555", label=f"Cada grupo de prueba (n={n_g})")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(S, "figuras", "fig_16g_resultados.png"), dpi=200); plt.close(fig)

C = pd.read_parquet(os.path.join(S, "16g_02_casos.parquet")); C = C[C["usable"]]
g = C.groupby(["grupo", "canal"]).agg(sync=("y_sync", "sum"), asyn=("y_async", "sum")).reset_index()
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, canal in zip(axs, ("cuerpo", "cara")):
    d = g[g["canal"] == canal].set_index("grupo")
    ax.bar(d.index, d["sync"], color="#2a6f97", label="Síncrono"); ax.bar(d.index, d["asyn"], bottom=d["sync"], color="#e07a1f", label="Asíncrono")
    ax.set_title(f"Canal {canal}"); ax.set_xlabel("Grupo"); ax.tick_params(axis="x", labelsize=8)
axs[0].set_ylabel("Casos positivos (pares-segmento)"); axs[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(S, "figuras", "fig_16g_casos.png"), dpi=200); plt.close(fig)
p("\nFiguras: salidas_16g/figuras/fig_16g_resultados.png y fig_16g_casos.png")
log.guardar(os.path.join(S, "16g_06_resumen.txt"))
