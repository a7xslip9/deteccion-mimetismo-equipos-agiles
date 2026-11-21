"""
16g_11_comparacion_pareada.py   (comparacion pareada de las variantes con el caso sin sinteticos)

Contexto: los intervalos de confianza de cada variante (16g_09 y 16g_10) se superponen con los del caso sin sinteticos (16g_05), asi que
que una variante pase de 2 veces el azar no basta para afirmar que mejora. Este script compara cada variante con el caso sin sinteticos
sobre LOS MISMOS casos (misma sesion, segmento, pareja y canal) y estima, con bootstrap por bloques pareado (bloques de 10 segmentos,
500 remuestreos), la DIFERENCIA de PR-AUC (variante menos base) y su intervalo de 95 %. Si el intervalo de la diferencia no incluye 0,
el cambio (mejora o empeoramiento) no se explica por el muestreo de los casos de prueba. Limite: este bootstrap remuestrea los casos de prueba,
no la variabilidad de entrenar de nuevo el modelo (semilla del bosque) ni la de los grupos.
Tambien compara algunos pares de variantes (mas sinteticos contra menos, representacion fina contra base).

Entradas (salidas_16g): 16g_05_puntajes_pareja.parquet (esquema logo, caracteristicas pose + manos), 16g_09_puntajes_pareja.parquet,
          16g_10_base_puntajes_pareja.parquet, 16g_10_fina_puntajes_pareja.parquet (las que existan)
Salidas:  salidas_16g/16g_11_comparacion_pareada.csv y 16g_11_comparacion_pareada.txt
Uso:      python 16g_11_comparacion_pareada.py
Requiere: pandas, pyarrow, numpy, scikit-learn
"""
import os, argparse
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
import util16g as u

ap = argparse.ArgumentParser(); ap.add_argument("--salida", default=u.SALIDA); ap.add_argument("--remuestreos", type=int, default=500); ap.add_argument("--semilla", type=int, default=0)
a = ap.parse_args(); log = u.Log(); p = log.p
KEY = ["sesion", "seg", "pareja", "canal"]

def leer(nombre, etiqueta):
    ruta = os.path.join(a.salida, nombre)
    if not os.path.exists(ruta):
        p("No existe:", nombre); return {}
    d = pd.read_parquet(ruta)
    return {etiqueta(o): g.dropna(subset=["puntaje"]) for o, g in d.groupby("objetivo")}

V = {}
b5 = pd.read_parquet(os.path.join(a.salida, "16g_05_puntajes_pareja.parquet"))
V["base-0"] = b5[(b5["esquema"] == "logo") & (b5["caract"] == "pose + manos")].dropna(subset=["puntaje"])
V.update(leer("16g_09_puntajes_pareja.parquet", lambda o: f"base-{o}"))
V.update(leer("16g_10_base_puntajes_pareja.parquet", lambda o: f"base-{o}"))
V.update(leer("16g_10_fina_puntajes_pareja.parquet", lambda o: f"fina-{o}"))
p("Variantes (representacion-sinteticos por accion):", sorted(V, key=lambda s: (s.split("-")[0], int(s.split("-")[1]))))

def comparar(canal, va, vb):
    """Diferencia de PR-AUC (vb menos va) sobre los mismos casos, bootstrap por bloques pareado."""
    A = V[va][V[va]["canal"] == canal][KEY + ["y", "bloque", "puntaje"]]; B = V[vb][V[vb]["canal"] == canal][KEY + ["y", "puntaje"]]
    M = A.merge(B, on=KEY + ["y"], suffixes=("_a", "_b"))
    y = M["y"].values; sa = M["puntaje_a"].values; sb = M["puntaje_b"].values
    ids, inv = np.unique(M["bloque"].values, return_inverse=True)
    por = [np.where(inv == k)[0] for k in range(len(ids))]
    rng = np.random.default_rng(a.semilla); d = []
    for _ in range(a.remuestreos):
        sel = np.concatenate([por[k] for k in rng.choice(len(ids), len(ids))])
        if y[sel].sum():
            d.append(average_precision_score(y[sel], sb[sel]) - average_precision_score(y[sel], sa[sel]))
    d = np.array(d); prev = y.mean()
    pa, pb = average_precision_score(y, sa), average_precision_score(y, sb)
    return dict(canal=canal, a=va, b=vb, casos=len(M), prevalencia=prev, x_azar_a=pa / prev, x_azar_b=pb / prev, dif_pr_auc=pb - pa,
                dif_ic_inf=np.percentile(d, 2.5), dif_ic_sup=np.percentile(d, 97.5), prop_remuestreos_mejora=float((d > 0).mean()),
                cambio=("mejora" if np.percentile(d, 2.5) > 0 else "empeora" if np.percentile(d, 97.5) < 0 else "sin cambio claro"))

pares = [("base-0", v) for v in V if v != "base-0"]
extra = [("base-2000", "base-5000"), ("base-2000", "fina-2000"), ("base-1000", "fina-1000"), ("base-0", "fina-0")]
pares += [e for e in extra if e[0] in V and e[1] in V and e not in pares]
filas = []
prog = u.Progreso(len(pares) * 2, "Comparaciones pareadas: una unidad = un par de variantes en un canal", log)
for va, vb in pares:
    for canal in ("cuerpo", "cara"):
        prog.actualizar(f"{va} -> {vb} | {canal}")
        filas.append(comparar(canal, va, vb)); prog.actualizar(avance=1)
prog.cerrar()
R = pd.DataFrame(filas); R.to_csv(os.path.join(a.salida, "16g_11_comparacion_pareada.csv"), index=False)
pd.set_option("display.width", 250)
p("\nDIFERENCIA DE PR-AUC (b menos a) sobre los mismos casos; intervalo de 95 % por bootstrap por bloques pareado")
for canal in ("cuerpo", "cara"):
    p(f"\n[{canal}]"); p(R[R["canal"] == canal].drop(columns="canal").round(4).to_string(index=False))
log.guardar(os.path.join(a.salida, "16g_11_comparacion_pareada.txt"))
