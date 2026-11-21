"""
16g_02_construir_casos.py   (MS1/MS3: casos y etiquetas; equivale al script 04)
Una fila por segmento de 2 s x pareja (6, sin direccion) x canal (cuerpo, cara).
Positivo = hay mimetismo anotado entre esas dos personas en ese canal (sincrono 'a=b' o asincrono 'a->b').
'usable' = ambas personas con datos validos en el segmento y canal (segun 16g_01).
Ademas informa cuantos positivos tiene cada grupo: un grupo sin positivos en un canal no se puede usar como grupo de prueba.

Uso (desde la carpeta scripts16g):   python 16g_02_construir_casos.py     (o con --grupos 01 15 para probar)
Entradas: Root/mimicry_indicators.xlsx, salidas_16g/16g_01_cobertura_<cod>.parquet
Salidas:  salidas_16g/16g_02_casos.parquet, 16g_02_construir_casos.txt
"""
import itertools, os, re
import pandas as pd
import util16g as u

a = u.parser("Casos y etiquetas (16 grupos)", root=False).parse_args()
os.makedirs(a.salida, exist_ok=True)
log = u.Log(); p = log.p
ann = u.leer_anotaciones(a.anot)
ses = u.sesiones_anotadas(ann, a.grupos)
PAREJAS = list(itertools.combinations([1, 2, 3, 4], 2))
PAT = re.compile(r"([1-4])\s*(=|->)\s*([1-4])")

def eventos(celda):
    res = []
    for m in PAT.finditer("" if pd.isna(celda) else str(celda)):
        i, tipo, j = int(m.group(1)), m.group(2), int(m.group(3))
        res.append((tuple(sorted((i, j))), "sync" if tipo == "=" else "async"))
    return res

filas = []; sin = []
prog = u.Progreso(len(ses), "Construyendo casos", log)
for cod in ses:
    prog.actualizar(f"sesion {cod}")
    ruta = os.path.join(a.salida, f"16g_01_cobertura_{cod}.parquet")
    if not os.path.exists(ruta):
        sin.append(cod); prog.actualizar(avance=1); continue
    cov = pd.read_parquet(ruta)
    ok = {(r.track_id, r.seg): (bool(r.cuerpo_ok), bool(r.cara_ok)) for r in cov.itertuples()}
    d = ann[ann["Grupo"].str.startswith(cod)].copy()
    d["seg"] = (pd.to_timedelta(d["time_start"]).dt.total_seconds() // 2).astype(int)
    for r in d.itertuples():
        for canal, celda, k in (("cuerpo", r.body_mimicry, 0), ("cara", r.face_mimicry, 1)):
            ev = eventos(celda)
            for (i, j) in PAREJAS:
                tipos = [t for (pr, t) in ev if pr == (i, j)]
                oki = ok.get((i, r.seg), (False, False))[k]; okj = ok.get((j, r.seg), (False, False))[k]
                filas.append(dict(sesion=cod, grupo=cod[:2], seg=r.seg, pareja=f"{i}-{j}", canal=canal,
                                  y=int(len(tipos) > 0), y_sync=int("sync" in tipos), y_async=int("async" in tipos),
                                  usable=bool(oki and okj)))
    prog.actualizar(avance=1)
prog.cerrar()
if sin:
    p("AVISO: sin series preparadas (falta correr 16g_01 para):", " ".join(sin))
C = pd.DataFrame(filas)
C.to_parquet(os.path.join(a.salida, "16g_02_casos.parquet"), index=False)

p("CASOS = segmentos anotados x 6 parejas x 2 canales")
p("sesiones incluidas:", C["sesion"].nunique(), "| grupos:", C["grupo"].nunique())
p("total de casos:", len(C), "| usables:", int(C["usable"].sum()))
U = C[C["usable"]]
t = U.groupby("canal").agg(casos=("y", "size"), positivos=("y", "sum"), sync=("y_sync", "sum"), asyn=("y_async", "sum"))
t["prevalencia_%"] = (t["positivos"] / t["casos"] * 100).round(2)
p("\nCasos usables por canal:"); p(t.to_string())
g = U.groupby(["grupo", "canal"]).agg(casos=("y", "size"), pos=("y", "sum")).unstack("canal")
g[("prev_%", "cuerpo")] = (g[("pos", "cuerpo")] / g[("casos", "cuerpo")] * 100).round(2)
g[("prev_%", "cara")] = (g[("pos", "cara")] / g[("casos", "cara")] * 100).round(2)
p("\nPor grupo (casos, positivos, prevalencia %):"); p(g.to_string())
bajos = [(gr, c) for gr in g.index for c in ("cuerpo", "cara") if g.loc[gr, ("pos", c)] < 10]
if bajos:
    p("\nAVISO: grupos con menos de 10 positivos en un canal (su resultado por grupo sera muy inestable):", bajos)
p("\nPositivos que se pierden por datos no validos:", int(((C["y"] == 1) & ~C["usable"]).sum()))
log.guardar(os.path.join(a.salida, "16g_02_construir_casos.txt"))
