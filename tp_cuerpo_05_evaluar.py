"""
tp_cuerpo_05_evaluar.py        (CRISP-DM, fase 5: EVALUACION; canal CUERPO)
Evalua los puntajes del script 04 y responde, con estadistica y sin un umbral arbitrario:
  (a) ¿cada modelo detecta mimetismo corporal mejor que el azar?
  (b) ¿cuanto mejor?  (tamano del efecto, con intervalo de confianza)
  (c) ¿un modelo es mejor que otro, o "todos los puntos" es mejor que los 17 descriptores?

POR QUE NO SE USA UN MULTIPLO FIJO DEL AZAR (reemplaza el antiguo criterio "PR-AUC >= 2 x prevalencia", D-20):
  ese 2 era una decision propia sin respaldo en la literatura, y un multiplo fijo ignora la incertidumbre: con pocos positivos
  un 2x puede ser ruido, y con muchos un 1,3x puede ser real. Aqui el criterio es estadistico y tiene fuente:
  - METRICA: PR-AUC (average precision), adecuada con clases muy desbalanceadas (Saito y Rehmsmeier, 2015; Davis y Goadrich, 2006).
    Su linea base es la prevalencia (el azar). Se reporta tambien el "skill" = (AP - prevalencia) / (1 - prevalencia): 0 = azar,
    1 = perfecto (forma general de un skill score respecto de una referencia; Wilks, 2011). Y la razon AP/prevalencia ("veces el azar").
  - VALIDACION: un grupo fuera (16 pruebas); cada grupo es una unidad independiente de evaluacion (Roberts et al., 2017;
    Demsar, 2006). La cifra principal es el promedio de las pruebas, no un puntaje unico de todos los casos mezclados.
  - ¿MEJOR QUE EL AZAR? Prueba de permutacion (Ojala y Garriga, 2010). Como las etiquetas son series de tiempo (un mimetismo dura
    varios segmentos seguidos), no se mezclan al azar: se DESPLAZAN CIRCULARMENTE dentro de cada serie (sesion x pareja), lo que
    conserva la duracion de los episodios y rompe su alineacion con los puntajes. Esta adaptacion es nuestra. Se corrige por
    comparaciones multiples con Holm (1979).
  - INCERTIDUMBRE: intervalo de confianza del 95 % por bootstrap en bloques de 10 segmentos consecutivos (Kuensch, 1989), que
    respeta la dependencia temporal; los modelos se comparan con las MISMAS remuestras (comparacion pareada).
  - CONSISTENCIA ENTRE GRUPOS: prueba de rangos con signo de Wilcoxon sobre las pruebas por grupo (Demsar, 2006) y cuantos
    grupos superan el azar.
  REGLA: un modelo "supera el azar" si p ajustado (Holm) < 0,05 Y el limite inferior del IC 95 % del skill supera el skill que se
  espera por puro azar (la media de la distribucion nula: con pocos positivos, el AP de un modelo al azar queda algo sobre la
  prevalencia). La MAGNITUD se reporta, no se usa como umbral. Si se necesitara un umbral de relevancia practica, debe fijarse con criterio del
  dominio (p. ej. el desacuerdo entre anotadores, que hoy no tenemos).

Uso (desde la carpeta scripts_tp):
    python tp_cuerpo_05_evaluar.py
    python tp_cuerpo_05_evaluar.py --perm 200 --boot 200        (mas rapido)
Entradas: salidas_tp_cuerpo/modelos_cuerpo/*.parquet
Salidas:  salidas_tp_cuerpo/tp_cuerpo_05_resumen_modelos.csv, tp_cuerpo_05_comparaciones.csv, tp_cuerpo_05_por_grupo.csv,
          tp_cuerpo_05_skill_por_grupo.png, tp_cuerpo_05_evaluar.txt
Requiere: numpy, pandas, scipy, scikit-learn, matplotlib
"""
import glob, itertools, os, re
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score
import tp_util as u

ap = u.parser("Evaluacion estadistica de cuerpo", root=False, anot=False, salida=u.SALIDA_CUERPO)
ap.add_argument("--perm", type=int, default=1000, help="permutaciones de la prueba de azar")
ap.add_argument("--boot", type=int, default=500, help="remuestras bootstrap")
ap.add_argument("--bloque", type=int, default=10, help="segmentos consecutivos por bloque del bootstrap")
ap.add_argument("--rep_ref", default="todos", help="representacion de la tabla principal")
a = ap.parse_args()
log = u.Log(); p = log.p
rng = np.random.default_rng(12345)

# ------------------------------------------------------------------------------------------------ carga
rutas = sorted(glob.glob(os.path.join(a.salida, "modelos_cuerpo", "*.parquet")))
if not rutas:
    raise SystemExit("No hay puntajes en salidas_tp_cuerpo/modelos_cuerpo. Ejecuta primero el script 04.")
PAT = re.compile(r"^(.+?)__(.+?)__s(\d+)__g(\d+)\.parquet$")
DATOS = {}      # (modelo, rep) -> {grupo: [puntajes por semilla]}
BASE = {}       # grupo -> DataFrame de claves y etiqueta
for r in rutas:
    m = PAT.match(os.path.basename(r))
    if not m:
        continue
    modelo, rep, sem, g = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    d = pd.read_parquet(r)
    if g not in BASE:
        BASE[g] = d[["sesion", "seg", "pareja", "y"]].reset_index(drop=True)
    elif not BASE[g][["sesion", "seg", "pareja", "y"]].equals(d[["sesion", "seg", "pareja", "y"]].reset_index(drop=True)):
        raise SystemExit(f"Los casos del grupo {g} no coinciden entre modelos ({os.path.basename(r)}). Vuelve a correr el script 04 con los mismos scripts 02 y 03.")
    DATOS.setdefault((modelo, rep), {}).setdefault(g, []).append(d["score"].to_numpy(float))
CFG = sorted(DATOS, key=lambda k: (k[1] != a.rep_ref, k[1], ["energia", "sincronia", "rf", "minirocket", "transformer"].index(k[0]) if k[0] in ["energia", "sincronia", "rf", "minirocket", "transformer"] else 9))
FOLDS = sorted(set.intersection(*[set(DATOS[c]) for c in CFG]))
perdidos = sorted(set(BASE) - set(FOLDS))
p(f"Configuraciones: {len(CFG)} | grupos de prueba comunes a todas: {len(FOLDS)} ({' '.join(FOLDS)})")
if perdidos:
    p("AVISO: estos grupos no estan completos en todas las configuraciones y se excluyen de la evaluacion:", perdidos)
nombre = lambda c: f"{c[0]} [{c[1]}]"

def holm(pv):
    """Correccion de Holm (1979): p ajustados para una lista de p-valores."""
    pv = np.asarray(pv, float); m_ = len(pv); orden = np.argsort(pv); aj = np.empty(m_); mx = 0.0
    for rank, i in enumerate(orden):
        mx = max(mx, (m_ - rank) * pv[i]); aj[i] = min(mx, 1.0)
    return aj

# ------------------------------------------------------------------------------------------------ AP exacto y rapido
class APRapido:
    """AP identico al de scikit-learn (con empates) pero reutilizable con distintas etiquetas: ordena una sola vez."""
    def __init__(self, score):
        self.o = np.argsort(-score, kind="mergesort"); s = score[self.o]
        self.fin = np.r_[np.where(s[1:] != s[:-1])[0], len(s) - 1]; self.n = self.fin + 1
    def __call__(self, y):
        yo = y[self.o]; npos = yo.sum()
        if npos == 0:
            return np.nan
        ctp = np.cumsum(yo)[self.fin]
        return float((ctp / self.n * np.diff(np.r_[0, ctp])).sum() / npos)

x = np.random.default_rng(0); _s = x.integers(0, 4, 500).astype(float); _y = (x.random(500) < 0.1).astype(int)
assert abs(APRapido(_s)(_y) - average_precision_score(_y, _s)) < 1e-9, "AP rapido no coincide con scikit-learn"

Y = {g: BASE[g]["y"].to_numpy(int) for g in FOLDS}
PREV = {g: Y[g].mean() for g in FOLDS}
APR = {c: {g: [APRapido(s) for s in DATOS[c][g]] for g in FOLDS} for c in CFG}
skill = lambda ap_, pr: (ap_ - pr) / (1 - pr)

def valores_observados(c):
    ap_g = {g: float(np.mean([f(Y[g]) for f in APR[c][g]])) for g in FOLDS}
    return ap_g, {g: skill(ap_g[g], PREV[g]) for g in FOLDS}

OBS = {c: valores_observados(c) for c in CFG}

# ------------------------------------------------------------------------------------------------ series para la permutacion y bloques para el bootstrap
SERIES = {}; BLOQUES = {}
for g in FOLDS:
    d = BASE[g]
    SERIES[g] = []
    for _, ix in d.groupby(["sesion", "pareja"]).indices.items():
        ix = np.asarray(ix); SERIES[g].append(ix[np.argsort(d["seg"].to_numpy()[ix], kind="mergesort")])   # serie ordenada por segmento
    b = (d["sesion"] + "_" + (d["seg"] // a.bloque).astype(str)).to_numpy()
    BLOQUES[g] = [np.where(b == k)[0] for k in np.unique(b)]

def y_desplazado(g):
    yp = np.empty_like(Y[g])
    for ix in SERIES[g]:
        k = int(rng.integers(1, len(ix))) if len(ix) > 1 else 0
        yp[ix] = np.roll(Y[g][ix], k)
    return yp

# ------------------------------------------------------------------------------------------------ prueba de permutacion
p("\n" + "=" * 90); p("1. ¿SUPERA CADA CONFIGURACION AL AZAR?  (permutacion con desplazamiento circular + bootstrap en bloques)"); p("=" * 90)
nulo = {c: np.empty(a.perm) for c in CFG}
prog = u.Progreso(a.perm, "Prueba de permutacion", log)
for k in range(a.perm):
    ypg = {g: y_desplazado(g) for g in FOLDS}
    for c in CFG:
        sk = [skill(np.mean([f(ypg[g]) for f in APR[c][g]]), PREV[g]) for g in FOLDS]
        nulo[c][k] = np.nanmean(sk)
    if k % 10 == 0:
        prog.actualizar(f"permutacion {k + 1}/{a.perm}", avance=10)
prog.cerrar()

# ------------------------------------------------------------------------------------------------ bootstrap en bloques (mismas remuestras para todos: comparacion pareada)
BOOT = np.full((a.boot, len(CFG)), np.nan)
prog = u.Progreso(a.boot, "Bootstrap en bloques", log)
for b in range(a.boot):
    sk = {c: [] for c in CFG}
    for g in FOLDS:
        nb = len(BLOQUES[g]); sel = np.concatenate([BLOQUES[g][i] for i in rng.integers(0, nb, nb)])
        ys = Y[g][sel]
        if ys.sum() == 0:
            continue
        pr = ys.mean()
        for c in CFG:
            sk[c].append(skill(np.mean([average_precision_score(ys, s[sel]) for s in DATOS[c][g]]), pr))
    for j, c in enumerate(CFG):
        BOOT[b, j] = np.mean(sk[c]) if sk[c] else np.nan
    if b % 10 == 0:
        prog.actualizar(f"remuestra {b + 1}/{a.boot}", avance=10)
prog.cerrar()

filas = []
for j, c in enumerate(CFG):
    ap_g, sk_g = OBS[c]
    apv = np.array([ap_g[g] for g in FOLDS]); skv = np.array([sk_g[g] for g in FOLDS]); prv = np.array([PREV[g] for g in FOLDS])
    sk_macro = float(np.nanmean(skv)); p_perm = (1 + np.sum(nulo[c] >= sk_macro)) / (a.perm + 1)
    lo, hi = np.nanpercentile(BOOT[:, j], [2.5, 97.5])
    try:
        pw = wilcoxon(skv, alternative="greater").pvalue
    except ValueError:
        pw = np.nan
    seeds = len(DATOS[c][FOLDS[0]])
    sd_sem = np.nan
    if seeds > 1:
        por_sem = [np.nanmean([skill(APR[c][g][s](Y[g]), PREV[g]) for g in FOLDS]) for s in range(seeds)]
        sd_sem = float(np.std(por_sem, ddof=1))
    filas.append(dict(modelo=c[0], rep=c[1], semillas=seeds, grupos=len(FOLDS), pr_auc_prom=apv.mean(), prevalencia_prom=prv.mean(),
                      x_azar_prom=float(np.mean(apv / prv)), skill_prom=sk_macro, skill_ic_inf=lo, skill_ic_sup=hi, sd_entre_semillas=sd_sem,
                      skill_nulo=float(np.mean(nulo[c])), grupos_sobre_azar=int((apv > prv).sum()), p_permutacion=p_perm, p_wilcoxon_grupos=pw))
T = pd.DataFrame(filas)
# Holm sobre los p de permutacion (todas las configuraciones evaluadas)
T["p_holm"] = holm(T["p_permutacion"])
T["supera_azar"] = (T["p_holm"] < 0.05) & (T["skill_ic_inf"] > T["skill_nulo"])
T.to_csv(os.path.join(a.salida, "tp_cuerpo_05_resumen_modelos.csv"), index=False)
vis = T.copy(); vis["modelo"] = vis["modelo"] + " [" + vis["rep"] + "]"
cols = ["modelo", "semillas", "pr_auc_prom", "prevalencia_prom", "x_azar_prom", "skill_prom", "skill_ic_inf", "skill_ic_sup", "skill_nulo", "grupos_sobre_azar", "p_holm", "supera_azar"]
p(vis[cols].round(3).to_string(index=False))
p(f"\n(promedio de {len(FOLDS)} pruebas, una por grupo. skill 0 = azar, 1 = perfecto. IC 95 % por bootstrap en bloques de {a.bloque} segmentos; "
  f"p por permutacion con {a.perm} desplazamientos, corregido con Holm.)")
p("Lectura: 'skill_nulo' es el skill que se espera por puro azar (media de la distribucion de permutacion). 'supera_azar' exige p ajustado < 0,05")
p("Y que el limite inferior del IC supere 'skill_nulo'. 'x_azar_prom' (AP/prevalencia) se informa solo como referencia, no como umbral.")

# ------------------------------------------------------------------------------------------------ comparaciones
p("\n" + "=" * 90); p("2. COMPARACIONES PAREADAS (diferencia de skill; IC por las mismas remuestras; Wilcoxon sobre las pruebas por grupo)"); p("=" * 90)
idx = {c: j for j, c in enumerate(CFG)}
pares = []
for rep in sorted({c[1] for c in CFG}):
    ref = ("rf", rep)
    for m in ("minirocket", "transformer"):
        if (m, rep) in idx and ref in idx:
            pares.append(((m, rep), ref, f"{m} contra la referencia rf (misma representacion: {rep})"))
    if ("transformer", rep) in idx and ("minirocket", rep) in idx:      # decision del modelo definitivo: Transformer contra MiniRocket
        pares.append((("transformer", rep), ("minirocket", rep), f"transformer contra minirocket (misma representacion: {rep})"))
    for m in ("rf", "minirocket", "transformer"):
        for b in ("sincronia", "energia"):
            if (m, rep) in idx and (b, rep) in idx:
                pares.append(((m, rep), (b, rep), f"{m} contra la base sin aprendizaje '{b}' ({rep})"))
for m in sorted({c[0] for c in CFG}):
    for r1, r2, txt in (("todos", "descriptores", "TODOS los puntos contra 17 descriptores"),
                        ("seleccion", "descriptores", "puntos SELECCIONADOS contra 17 descriptores"),
                        ("seleccion", "todos", "puntos SELECCIONADOS contra TODOS los puntos")):
        if (m, r1) in idx and (m, r2) in idx:
            pares.append(((m, r1), (m, r2), f"{m}: {txt}"))
fc = []
for c1, c2, desc in pares:
    d_g = np.array([OBS[c1][1][g] - OBS[c2][1][g] for g in FOLDS])
    dif = BOOT[:, idx[c1]] - BOOT[:, idx[c2]]
    lo, hi = np.nanpercentile(dif, [2.5, 97.5])
    try:
        pw = wilcoxon(d_g).pvalue
    except ValueError:
        pw = np.nan
    fc.append(dict(comparacion=desc, dif_skill=float(np.nanmean(d_g)), ic_inf=lo, ic_sup=hi, grupos_a_favor=int((d_g > 0).sum()), grupos=len(FOLDS), p_wilcoxon=pw))
if fc:
    F = pd.DataFrame(fc); F["p_holm"] = holm(F["p_wilcoxon"].fillna(1.0))
    F.to_csv(os.path.join(a.salida, "tp_cuerpo_05_comparaciones.csv"), index=False)
    p(F.round(3).to_string(index=False))
    p("\nUna diferencia es respaldada si el IC pareado no incluye 0 y el Wilcoxon por grupos (corregido con Holm) tambien la apoya.")
    p("Con 16 pruebas el Wilcoxon tiene poco poder: el IC pareado por bootstrap es la evidencia principal y el Wilcoxon, la de consistencia entre grupos.")
else:
    p("(no hay pares comparables todavia: faltan modelos o representaciones)")

# ------------------------------------------------------------------------------------------------ por grupo y figura
G = pd.DataFrame({nombre(c): {g: OBS[c][1][g] for g in FOLDS} for c in CFG})
G.insert(0, "prevalencia", pd.Series(PREV)); G.round(3).to_csv(os.path.join(a.salida, "tp_cuerpo_05_por_grupo.csv"))
p("\n" + "=" * 90); p("3. SKILL POR GRUPO DE PRUEBA (0 = azar)"); p("=" * 90); p(G.round(2).to_string())
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(CFG)), 4.5))
    datos = [[OBS[c][1][g] for g in FOLDS] for c in CFG]
    ax.boxplot(datos, showmeans=True); ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xticks(range(1, len(CFG) + 1)); ax.set_xticklabels([nombre(c).replace(" [", "\n[") for c in CFG], fontsize=8)
    ax.set_ylabel("skill por grupo de prueba (0 = azar)"); ax.set_title("Mimetismo corporal: desempeño por grupo, un grupo fuera")
    fig.tight_layout(); fig.savefig(os.path.join(a.salida, "tp_cuerpo_05_skill_por_grupo.png"), dpi=130)
except Exception as e:
    p("(no se pudo dibujar la figura:", e, ")")
p("\nLimitaciones que deben ir en el informe: (1) la etiqueta es una regla del Excel, no gestos verificados; (2) la pose y las manos de MediaPipe son estimaciones con ruido (los dedos "
  "mas que los hombros) y los puntos ocultos se tratan como faltantes; (3) la normalizacion por persona usa toda la sesion de esa persona (no es en linea); (4) los modelos no tuvieron "
  "busqueda de hiperparametros; (5) hay 16 grupos y pocos positivos por grupo.")
log.guardar(os.path.join(a.salida, "tp_cuerpo_05_evaluar.txt"))
