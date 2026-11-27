"""
tp_cara_03_construir_casos.py        (CRISP-DM, fase 3: PREPARACION DE LOS DATOS, parte 2; canal CARA)
Define QUE se predice: un caso = (sesion, segmento de 2 s, pareja de personas). Etiqueta y = 1 si el Excel marca mimetismo
FACIAL entre esas dos personas en ese segmento (columna face_mimicry: 'i=j' sincrono, 'i->j' asincrono; sin direccion).
Tambien decide que casos son USABLES y deja los indices que usa el script 04 para armar las ventanas.

Decisiones (detalle y fuentes en DECISIONES_cara.md):
  D-12  Unidad de prediccion = pareja x segmento (6 parejas por segmento, sin direccion): asi viene anotado el Excel.
  D-13  Ventana de entrada = segmento anotado + el anterior (4 s): la respuesta de un mimetismo puede llegar despues del
        gesto original (hay anotaciones asincronas 'i->j'); MAHNOB define el mimetismo como la respuesta dentro de 4 s
        del gesto original (Bilakhia et al., 2015).
  D-05  Caso usable = ambas personas con >= 70 % de cuadros validos en el segmento anotado y >= 50 % en el anterior.
  D-14  Validacion = un grupo fuera (leave-one-group-out): se entrena con 15 grupos y se prueba con el que queda. Evita
        que segmentos vecinos del mismo grupo caigan en entrenamiento y prueba (Roberts et al., 2017).

Uso (desde la carpeta scripts_tp):
    python tp_cara_03_construir_casos.py          (o con --grupos 01 15 para probar)
Entradas: Root/mimicry_indicators.xlsx, salidas_tp/tp_cara_02_<sesion>.npz
Salidas:  salidas_tp/tp_cara_03_casos.parquet, tp_cara_03_personas.parquet, tp_cara_03_construir_casos.txt
"""
import itertools, os, re
import numpy as np, pandas as pd
import tp_util as u

ap = u.parser("Casos y etiquetas de cara", root=False)
ap.add_argument("--cob_actual", type=float, default=u.COB_ACTUAL)
ap.add_argument("--cob_previo", type=float, default=u.COB_PREVIO)
ap.add_argument("--series", default=None, help="carpeta con los tp_*_02_<sesion>.npz (por defecto: la de --salida); sirve para el analisis de sensibilidad")
a = ap.parse_args()
os.makedirs(a.salida, exist_ok=True)
SERIES = a.series or a.salida
log = u.Log(); p = log.p
ann = u.leer_anotaciones(a.anot)
ses = [s for s in u.sesiones_anotadas(ann, a.grupos) if os.path.exists(os.path.join(SERIES, f"tp_cara_02_{s}.npz"))]
faltan = [s for s in u.sesiones_anotadas(ann, a.grupos) if s not in ses]
if faltan:
    p("AVISO: sin series preparadas (falta correr tp_cara_02 para):", " ".join(faltan))
PAREJAS = list(itertools.combinations([1, 2, 3, 4], 2))
PAT = re.compile(r"([1-4])\s*(=|->)\s*([1-4])")


def eventos(celda):
    res = []
    for m in PAT.finditer("" if pd.isna(celda) else str(celda)):
        res.append((tuple(sorted((int(m.group(1)), int(m.group(3))))), "sync" if m.group(2) == "=" else "async"))
    return res


# ---- tabla de personas-segmento, en el MISMO orden en que el script 04 concatena los arrays ----
filas, base = [], 0
for cod in ses:
    z = np.load(os.path.join(SERIES, f"tp_cara_02_{cod}.npz"))
    n = len(z["seg"])
    filas.append(pd.DataFrame(dict(sesion=cod, track=z["track"].astype(int), seg=z["seg"].astype(int), cov=z["cov"], fila=np.arange(base, base + n))))
    base += n
PER = pd.concat(filas, ignore_index=True)
PER.to_parquet(os.path.join(a.salida, "tp_cara_03_personas.parquet"), index=False)
look = {(r.sesion, r.track, r.seg): (r.fila, r.cov) for r in PER.itertuples()}
p(f"Personas-segmento disponibles: {len(PER)} (sesiones: {len(ses)})")

# ---- casos ----
casos = []
ann = ann[ann["Grupo"].str[:3].isin(ses)].copy()
ann["seg"] = (pd.to_timedelta(ann["time_start"]).dt.total_seconds() // u.SEG_S).astype(int)
prog = u.Progreso(len(ann), "Construyendo casos", log)
for k, r in enumerate(ann.itertuples()):
    cod = r.Grupo[:3]
    ev = eventos(r.face_mimicry)
    for (i, j) in PAREJAS:
        tipos = [t for (pr, t) in ev if pr == (i, j)]
        ci, cj = look.get((cod, i, r.seg)), look.get((cod, j, r.seg))
        pi, pj = look.get((cod, i, r.seg - 1)), look.get((cod, j, r.seg - 1))
        usable = bool(ci and cj and pi and pj and ci[1] >= a.cob_actual and cj[1] >= a.cob_actual
                      and pi[1] >= a.cob_previo and pj[1] >= a.cob_previo)
        casos.append(dict(sesion=cod, grupo=cod[:2], seg=int(r.seg), pareja=f"{i}-{j}", y=int(len(tipos) > 0),
                          y_sync=int("sync" in tipos), y_async=int("async" in tipos), usable=usable,
                          fila_i=ci[0] if ci else -1, fila_j=cj[0] if cj else -1, fila_pi=pi[0] if pi else -1, fila_pj=pj[0] if pj else -1))
    if k % 200 == 0:
        prog.actualizar(f"fila {k}/{len(ann)}", avance=200 if k else 0)
prog.cerrar()
C = pd.DataFrame(casos)
C.to_parquet(os.path.join(a.salida, "tp_cara_03_casos.parquet"), index=False)

U = C[C["usable"]]
p("\nCASOS = segmentos anotados x 6 parejas (canal cara)")
p(f"sesiones: {C['sesion'].nunique()} | grupos: {C['grupo'].nunique()} | casos totales: {len(C)} | usables: {len(U)} ({len(U) / max(len(C), 1) * 100:.1f} %)")
p(f"positivos totales: {int(C['y'].sum())} | positivos usables: {int(U['y'].sum())} (prevalencia {U['y'].mean() * 100:.2f} %) | "
  f"sincronos: {int(U['y_sync'].sum())} | asincronos: {int(U['y_async'].sum())}")
p(f"positivos que se pierden por datos no validos: {int(((C['y'] == 1) & ~C['usable']).sum())}")
G = U.groupby("grupo").agg(casos=("y", "size"), positivos=("y", "sum")); G["prevalencia_%"] = (G["positivos"] / G["casos"] * 100).round(2)
p("\nPor grupo (solo casos usables):"); p(G.to_string())
sin = list(G.index[G["positivos"] == 0]); pocos = list(G.index[(G["positivos"] > 0) & (G["positivos"] < 10)])
if sin:
    p("AVISO: grupos SIN positivos usables (no pueden ser grupo de prueba; si se usan solo en entrenamiento):", sin)
if pocos:
    p("AVISO: grupos con menos de 10 positivos (su resultado individual sera muy inestable):", pocos)
p(f"\nGrupos que se podran usar como prueba (>= 1 positivo): {int((G['positivos'] > 0).sum())} de {len(G)}")
log.guardar(os.path.join(a.salida, "tp_cara_03_construir_casos.txt"))
