"""
16g_07_diagnostico_actividad.py   (diagnostico; equivale al script 07, con los 16 grupos)
Pregunta: el mimetismo anotado, es lo mismo que "dos personas con la misma accion anotada al mismo tiempo"?
Regla sin aprendizaje: puntaje = 1 si ambas personas tienen la misma accion (distinta de 'none') en el segmento.
NO es un detector (usa las anotaciones de accion, que no existen en un despliegue real): sirve para conocer el TECHO del
enfoque en dos etapas y para entender por que el canal cuerpo rinde menos que el canal cara.
Ademas informa, por accion, cuantas veces que dos personas la hacen juntas hay mimetismo anotado en cada canal.

Uso (desde la carpeta scripts16g):   python 16g_07_diagnostico_actividad.py     (rapido)
Entradas: Root/mimicry_indicators.xlsx, salidas_16g/16g_02_casos.parquet
Salidas:  salidas_16g/16g_07_diagnostico_actividad.txt, 16g_07_por_accion.csv
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
import util16g as u

a = u.parser("Diagnostico de actividad (16 grupos)", root=False).parse_args()
log = u.Log(); p = log.p
ann = u.leer_anotaciones(a.anot)
ann["sesion"] = ann["Grupo"].str[:3]
ann["seg"] = (pd.to_timedelta(ann["time_start"]).dt.total_seconds() // 2).astype(int)
for k in range(1, 5):
    ann[f"s{k}"] = ann[f"speaker{k}"].fillna("none").str.strip().str.lower()
casos = pd.read_parquet(os.path.join(a.salida, "16g_02_casos.parquet"))
casos = casos[casos["usable"]].copy()
acc = ann.drop_duplicates(["sesion", "seg"]).set_index(["sesion", "seg"])[[f"s{k}" for k in range(1, 5)]]
idx = acc.index
fila = acc.reindex(pd.MultiIndex.from_arrays([casos["sesion"], casos["seg"]]))
i = casos["pareja"].str[0].astype(int).values; j = casos["pareja"].str[2].astype(int).values
S = fila.values
ai = S[np.arange(len(casos)), i - 1]; aj = S[np.arange(len(casos)), j - 1]
casos["accion"] = np.where((ai == aj) & (ai != "none"), ai, "")
casos["misma_accion"] = (casos["accion"] != "").astype(int)

p("Regla: misma accion anotada (distinta de 'none') en el mismo segmento\n")
filas = []
for canal in ("cuerpo", "cara"):
    d = casos[casos["canal"] == canal]
    for nombre, g in [("TODOS", d)] + [(f"grupo {k}", v) for k, v in d.groupby("grupo")]:
        y, s = g["y"].values, g["misma_accion"].values
        tp = int(((s == 1) & (y == 1)).sum()); fp = int(((s == 1) & (y == 0)).sum()); fn = int(((s == 0) & (y == 1)).sum())
        pa = average_precision_score(y, s) if y.sum() else np.nan
        p(f"[{canal}] {nombre}: casos={len(g)}, positivos={int(y.sum())}, prevalencia={y.mean():.3f} | PR-AUC={pa:.3f} "
          f"({pa / y.mean():.1f}x azar) | precision={tp / max(tp + fp, 1):.3f}, exhaustividad={tp / max(tp + fn, 1):.3f} | "
          f"positivos con misma accion={tp} de {tp + fn}; misma accion sin mimetismo={fp}")
    p("")

p("=" * 100); p("POR ACCION: cuando dos personas hacen la MISMA accion a la vez, en que % de los casos hay mimetismo anotado"); p("=" * 100)
for canal in ("cuerpo", "cara"):
    d = casos[(casos["canal"] == canal) & (casos["misma_accion"] == 1)]
    t = d.groupby("accion").agg(casos=("y", "size"), con_mimetismo=("y", "sum"))
    t["%_con_mimetismo"] = (t["con_mimetismo"] / t["casos"] * 100).round(1)
    t["canal"] = canal
    t = t.sort_values("con_mimetismo", ascending=False)
    filas.append(t.reset_index())
    p(f"\n[{canal}] total con misma accion={len(d)}, con mimetismo={int(d['y'].sum())} ({d['y'].mean() * 100:.1f} %)")
    p(t.drop(columns="canal").head(15).to_string())
pd.concat(filas).to_csv(os.path.join(a.salida, "16g_07_por_accion.csv"), index=False)
log.guardar(os.path.join(a.salida, "16g_07_diagnostico_actividad.txt"))
