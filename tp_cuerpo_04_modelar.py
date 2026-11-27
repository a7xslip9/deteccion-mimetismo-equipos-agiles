"""
tp_cuerpo_04_modelar.py        (CRISP-DM, fase 4: MODELADO; canal CUERPO)
Entrena y prueba, con la validacion "un grupo fuera" (D-14), los modelos que puntuan cada pareja y segmento con la
probabilidad de que haya mimetismo CORPORAL. Se puede correr con TODOS LOS PUNTOS (--rep todos), con los puntos seleccionados (--rep seleccion) o con los 17 descriptores
anteriores (--rep descriptores), para comparar las dos representaciones con la misma validacion (B-02).

ENTRADA de cada modelo (D-15): la ventana de 4 s de la pareja (segmento anterior + segmento anotado), en cada instante y canal:
        [ promedio de las dos personas , |diferencia entre las dos personas| ]
   El promedio dice cuanto se mueve la pareja; la diferencia dice cuanto se parecen. Es simetrica: el orden (i, j) no importa,
   porque la anotacion no tiene direccion. Las series ya vienen alineadas y normalizadas por persona (script 02).

MODELOS (--modelos):
  energia       Base SIN aprendizaje: actividad de la pareja = minimo de la energia de movimiento de las dos personas.
                Si un modelo no la supera, solo esta detectando "ambos se mueven", no mimetismo.            (D-16)
  sincronia     Base SIN aprendizaje: correlacion promedio, canal a canal, entre las series de las dos personas
                en la ventana (medida clasica de sincronia, Delaherche et al., 2012).                          (D-16)
  rf            Random Forest sobre promedio, desviacion y rango de cada canal de la ventana. Es la REFERENCIA
                solicitada (Breiman, 2001; Grinsztajn et al., 2022).                                            (D-17)
  minirocket    MiniRocket (Dempster et al., 2021; aeon): filtros convolucionales aleatorios + regresion logistica. (D-18)
  transformer   Codificador Transformer pequeno sobre la ventana (Zerveas et al., 2021), en PyTorch (GPU si hay). (D-19)

Los modelos con aprendizaje se ajustan SOLO con los 15 grupos de entrenamiento de cada prueba; ninguna decision depende
del grupo de prueba. No hay busqueda de hiperparametros (valores fijos y declarados), para no sobreajustar a estos grupos.

OPCIONES DE SENSIBILIDAD (D-28 / B-15): --ventana 2 (solo el segmento anotado), --sufijo ETIQUETA (nombre de la configuracion), --series CARPETA (series
de otra carpeta), --puntos RUTA.json (otra seleccion). Cada variante se corre con su propio --salida para no mezclar casos distintos.

Resumible: cada prueba (modelo x representacion x semilla x grupo) se guarda en salidas_tp_cuerpo/modelos_cuerpo/. Si el programa se
corta, vuelve a ejecutar el mismo comando y continua donde quedo (--rehacer para forzar).

Uso (desde la carpeta scripts_tp), ver LEEME_cuerpo.md para el orden completo:
    python tp_cuerpo_04_modelar.py --rep todos --modelos energia sincronia rf
    python tp_cuerpo_04_modelar.py --rep todos --modelos minirocket
    python tp_cuerpo_04_modelar.py --rep todos --modelos transformer
    python tp_cuerpo_04_modelar.py --rep seleccion --modelos energia sincronia rf
    python tp_cuerpo_04_modelar.py --rep descriptores --modelos energia sincronia rf minirocket transformer
    python tp_cuerpo_04_modelar.py --rep todos --modelos rf minirocket transformer --rapido     (prueba de humo)
Entradas: salidas_tp_cuerpo/tp_cuerpo_02_<sesion>.npz, tp_cuerpo_03_casos.parquet, tp_cuerpo_03_personas.parquet
Salidas:  salidas_tp_cuerpo/modelos_cuerpo/<modelo>__<rep>__s<semilla>__g<grupo>.parquet (puntajes por caso de prueba;
          con --rapido van a modelos_cuerpo_humo para no mezclarse con los reales),
          tp_cuerpo_04_<rep>_<modelos>_resultados.csv, tp_cuerpo_04_<rep>_modelar.txt
Requiere: numpy, pandas, scikit-learn, aeon (minirocket), torch (transformer)
"""
import os, time, traceback, warnings
import numpy as np, pandas as pd
import tp_util as u

warnings.filterwarnings("ignore")
ap = u.parser("Modelado de cuerpo, un grupo fuera", root=False, anot=False, salida=u.SALIDA_CUERPO)
ap.add_argument("--rep", choices=["todos", "seleccion", "descriptores"], default="todos",
                help="todos = 154 canales de pose y manos; seleccion = puntos elegidos por tp_cuerpo_03b (B-09); descriptores = 17 descriptores")
ap.add_argument("--modelos", nargs="+", default=["energia", "sincronia", "rf", "minirocket", "transformer"],
                choices=["energia", "sincronia", "rf", "minirocket", "transformer"])
ap.add_argument("--semillas", nargs="+", type=int, default=None, help="por defecto: 0 1 2 para transformer y 0 para los demas")
ap.add_argument("--kernels", type=int, default=10000, help="filtros de MiniRocket (D-18)")
ap.add_argument("--epocas", type=int, default=10, help="epocas del Transformer (D-19)")
ap.add_argument("--arboles", type=int, default=300, help="arboles del Random Forest (D-17)")
ap.add_argument("--lote_ventanas", type=int, default=1000, help="casos por bloque al armar ventanas (baja si falta memoria)")
ap.add_argument("--max_folds", type=int, default=None, help="solo para pruebas: limitar el numero de grupos de prueba")
ap.add_argument("--rapido", action="store_true", help="prueba de humo: pocos filtros, pocas epocas, pocos arboles, 2 grupos")
ap.add_argument("--rehacer", action="store_true")
ap.add_argument("--sufijo", default="", help="etiqueta que se agrega al nombre de la representacion en los archivos (analisis de sensibilidad)")
ap.add_argument("--ventana", type=int, choices=[2, 4], default=4,
                help="4 = segmento anterior + segmento anotado (D-13, por defecto); 2 = solo el segmento anotado (sensibilidad)")
ap.add_argument("--series", default=None, help="carpeta con los tp_*_02_<sesion>.npz (por defecto: la de --salida)")
ap.add_argument("--puntos", default=None, help="json de la seleccion de puntos (por defecto: el de la carpeta de series)")
a = ap.parse_args()
if a.rapido:
    a.kernels, a.epocas, a.arboles, a.max_folds = 840, 2, 30, 2
SEMILLA0 = 0
SERIES = a.series or a.salida
REP = a.rep + a.sufijo          # nombre de la configuracion en los archivos
PREV = (a.ventana == 4)         # True: la ventana incluye el segmento anterior
SUBCARPETA = "modelos_cuerpo_humo" if a.rapido else "modelos_cuerpo"      # la prueba de humo NO mezcla sus resultados con los reales
os.makedirs(os.path.join(a.salida, SUBCARPETA), exist_ok=True)
log = u.Log(); p = log.p

# ------------------------------------------------------------------------------------------------ datos
PER = pd.read_parquet(os.path.join(a.salida, "tp_cuerpo_03_personas.parquet"))
CAS = pd.read_parquet(os.path.join(a.salida, "tp_cuerpo_03_casos.parquet"))
CAS = CAS[CAS["usable"]].reset_index(drop=True)
if a.grupos:
    CAS = CAS[CAS["grupo"].isin([g.zfill(2) for g in a.grupos])].reset_index(drop=True)
clave = "X_desc" if a.rep == "descriptores" else "X_todos"
dt = np.float32 if a.rep == "descriptores" else np.float16
COLS = None
if a.rep == "seleccion":
    import json
    ruta_sel = a.puntos or os.path.join(SERIES, "tp_cuerpo_03b_puntos.json")
    if not os.path.exists(ruta_sel):
        raise SystemExit(f"Falta {ruta_sel}: corre antes tp_cuerpo_03b_seleccionar_puntos.py o indica --puntos")
    COLS = np.array(json.load(open(ruta_sel))["canales"])
ses = list(dict.fromkeys(PER["sesion"]))
P = None; base = 0
for cod in ses:                                              # mismo orden que en el script 03 (fila global)
    z = np.load(os.path.join(SERIES, f"tp_cuerpo_02_{cod}.npz"))[clave]
    if COLS is not None:
        z = z[:, :, COLS]
    if P is None:
        P = np.empty((len(PER), z.shape[1], z.shape[2]), dtype=dt)
    P[base:base + len(z)] = z; base += len(z)
assert base == len(PER), "el orden de las sesiones no coincide con tp_cuerpo_03_personas.parquet: vuelve a correr los scripts 02 y 03"
N_C = P.shape[2]; T1 = P.shape[1]
y = CAS["y"].to_numpy(int); grupo = CAS["grupo"].to_numpy(); sesion = CAS["sesion"].to_numpy(); seg = CAS["seg"].to_numpy(); pareja = CAS["pareja"].to_numpy()
CI, CJ, PI, PJ = (CAS[c].to_numpy(int) for c in ("fila_i", "fila_j", "fila_pi", "fila_pj"))
p(f"Representacion: {a.rep} | canales por persona: {N_C} | personas-segmento: {len(P)} | casos usables: {len(CAS)} "
  f"| positivos: {int(y.sum())} ({y.mean() * 100:.2f} %) | entrada del modelo: {2 * N_C} canales x {(2 if PREV else 1) * T1} instantes (ventana de {2 * (2 if PREV else 1)} s)")
GRUPOS = sorted(set(grupo))
GRUPOS_PRUEBA = [g for g in GRUPOS if y[grupo == g].sum() > 0 and y[grupo != g].sum() > 0]
if a.max_folds:
    GRUPOS_PRUEBA = GRUPOS_PRUEBA[:a.max_folds]
p(f"Grupos de prueba: {len(GRUPOS_PRUEBA)} de {len(GRUPOS)} ({' '.join(GRUPOS_PRUEBA)})")


def par(A, B):
    return np.concatenate([(A + B) / 2, np.abs(A - B)], axis=2)


def ventana(ix):
    """Ventanas de 4 s de los casos ix: casos x 2T instantes x 2C canales (float32). Se arman por bloques, nunca todas a la vez."""
    f = lambda k: P[k[ix]].astype(np.float32)
    act = par(f(CI), f(CJ))
    return np.concatenate([par(f(PI), f(PJ)), act], axis=1) if PREV else act


def por_bloques(ix, fun):
    return np.concatenate([fun(ix[k:k + a.lote_ventanas]) for k in range(0, len(ix), a.lote_ventanas)])


def resumen(X):                        # X: n x T x C  ->  n x 3C (promedio, desviacion, rango por canal)
    return np.concatenate([X.mean(1), X.std(1), X.max(1) - X.min(1)], axis=1)


# ------------------------------------------------------------------------------------------------ modelos
def puntaje_energia(ix):
    """Energia de movimiento por persona = promedio del cambio absoluto cuadro a cuadro en todos los canales."""
    def en(k):
        return np.abs(np.diff(P[k].astype(np.float32), axis=1)).mean(axis=(1, 2))
    def f(ix2):
        act = np.minimum(en(CI[ix2]), en(CJ[ix2]))
        return (act + np.minimum(en(PI[ix2]), en(PJ[ix2]))) / 2 if PREV else act
    return por_bloques(ix, f)


def puntaje_sincronia(ix):
    """Correlacion de Pearson canal a canal entre las dos personas en la ventana de 4 s, promediada sobre canales."""
    def f(ix2):
        if PREV:
            A = np.concatenate([P[PI[ix2]], P[CI[ix2]]], axis=1).astype(np.float32); B = np.concatenate([P[PJ[ix2]], P[CJ[ix2]]], axis=1).astype(np.float32)
        else:
            A = P[CI[ix2]].astype(np.float32); B = P[CJ[ix2]].astype(np.float32)
        A -= A.mean(1, keepdims=True); B -= B.mean(1, keepdims=True)
        den = np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))
        r = np.where(den > 1e-6, (A * B).sum(1) / np.maximum(den, 1e-6), 0.0)
        return r.mean(1)
    return por_bloques(ix, f)


RF_CACHE = {}
def rf_features():
    ruta = os.path.join(a.salida, f"tp_cuerpo_04_rf_features_{REP}_v{a.ventana}_{len(CAS)}.npy")
    if "F" not in RF_CACHE:
        fuentes = [os.path.join(a.salida, "tp_cuerpo_03_casos.parquet"), os.path.join(a.salida, "tp_cuerpo_03_personas.parquet")]
        if os.path.exists(ruta) and os.path.getmtime(ruta) > max(os.path.getmtime(f) for f in fuentes) and not a.rehacer:
            RF_CACHE["F"] = np.load(ruta, mmap_mode="r")
        else:
            p("  calculando resumenes de las ventanas para el Random Forest (una sola vez)...")
            F = por_bloques(np.arange(len(CAS)), lambda ix: resumen(ventana(ix)))
            np.save(ruta, F.astype(np.float32)); RF_CACHE["F"] = F
    return RF_CACHE["F"]


def ajustar_rf(tr, te, semilla):
    from sklearn.ensemble import RandomForestClassifier
    F = rf_features()
    m = RandomForestClassifier(n_estimators=a.arboles, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample",
                               n_jobs=-1, random_state=semilla)
    m.fit(np.asarray(F[tr]), y[tr])
    return m.predict_proba(np.asarray(F[te]))[:, 1]


def ajustar_minirocket(tr, te, semilla):
    from aeon.transformations.collection.convolution_based import MiniRocket
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(semilla)
    sub = np.sort(rng.choice(tr, min(3000, len(tr)), replace=False))        # D-18: los sesgos de los filtros usan una muestra del entrenamiento
    t = MiniRocket(n_kernels=a.kernels, random_state=semilla)
    t.fit(np.ascontiguousarray(ventana(sub).transpose(0, 2, 1)))
    tf = lambda ix: por_bloques(ix, lambda i2: np.asarray(t.transform(np.ascontiguousarray(ventana(i2).transpose(0, 2, 1)))).astype(np.float32))
    Ftr, Fte = tf(tr), tf(te)
    sc = StandardScaler().fit(Ftr)
    m = LogisticRegression(C=0.01, class_weight="balanced", max_iter=300)
    m.fit(sc.transform(Ftr), y[tr])
    return m.decision_function(sc.transform(Fte))


TR = {}
def ajustar_transformer(tr, te, semilla):
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if "P" not in TR:
        TR["P"] = torch.from_numpy(P).to(dev)
        TR["idx"] = [torch.from_numpy(k).to(dev) for k in (CI, CJ, PI, PJ)]
        TR["y"] = torch.tensor(y, dtype=torch.float32, device=dev)
    Pt, (ci, cj, pi, pj), yt = TR["P"], TR["idx"], TR["y"]
    torch.manual_seed(semilla); np.random.seed(semilla)
    D = 64

    class Red(nn.Module):
        def __init__(s, c_in, T):
            super().__init__()
            s.entrada = nn.Linear(c_in, D); s.pos = nn.Parameter(torch.zeros(1, T, D))
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 128, dropout=0.2, batch_first=True), 2)
            s.sal = nn.Linear(D, 1)
        def forward(s, x):                                   # x: lote x instantes x canales
            return s.sal(s.enc(s.entrada(x) + s.pos).mean(1)).squeeze(-1)

    def lote(ix):
        f = lambda k: Pt[k[ix]].float()
        pr = lambda A, B: torch.cat([(A + B) / 2, (A - B).abs()], dim=2)
        return torch.cat([pr(f(pi), f(pj)), pr(f(ci), f(cj))], dim=1) if PREV else pr(f(ci), f(cj))

    red = Red(2 * N_C, (2 if PREV else 1) * T1).to(dev)
    opt = torch.optim.AdamW(red.parameters(), lr=1e-3, weight_decay=1e-2)
    ytr = y[tr]; pw = torch.tensor(min((len(ytr) - ytr.sum()) / max(ytr.sum(), 1), 20.0), dtype=torch.float32, device=dev)
    perd = nn.BCEWithLogitsLoss(pos_weight=pw)
    trt = torch.from_numpy(tr).to(dev); tet = torch.from_numpy(te).to(dev)
    for ep in range(a.epocas):
        red.train(); perm = trt[torch.randperm(len(trt), device=dev)]
        for k in range(0, len(perm), 256):
            ix = perm[k:k + 256]; opt.zero_grad(); perd(red(lote(ix)), yt[ix]).backward()
            torch.nn.utils.clip_grad_norm_(red.parameters(), 1.0); opt.step()
    red.eval(); out = []
    with torch.no_grad():
        for k in range(0, len(tet), 512):
            out.append(red(lote(tet[k:k + 512])).cpu().numpy())
    return np.concatenate(out)


SIN_ENTRENAR = {"energia": puntaje_energia, "sincronia": puntaje_sincronia}
ENTRENADOS = {"rf": ajustar_rf, "minirocket": ajustar_minirocket, "transformer": ajustar_transformer}

# ------------------------------------------------------------------------------------------------ bucle de pruebas
def semillas_de(m):
    return a.semillas if a.semillas is not None else ([0, 1, 2] if m == "transformer" else [0])


unidades = [(m, s, g) for m in a.modelos for s in ([0] if m in SIN_ENTRENAR else semillas_de(m)) for g in GRUPOS_PRUEBA]
ruta_de = lambda m, s, g: os.path.join(a.salida, SUBCARPETA, f"{m}__{REP}__s{s}__g{g}.parquet")
pend = [x for x in unidades if a.rehacer or not os.path.exists(ruta_de(*x))]
p(f"Pruebas totales: {len(unidades)} | ya hechas: {len(unidades) - len(pend)} | por hacer: {len(pend)}")
try:
    import torch
    p("Dispositivo del Transformer:", "GPU " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (lento)")
except Exception:
    pass
cache_baselines = {}
fallas = []
if pend:
    prog = u.Progreso(len(pend), "Modelado: un grupo fuera", log)
    for k, (m, s, g) in enumerate(pend, 1):
        te = np.where(grupo == g)[0]; tr = np.where(grupo != g)[0]
        prog.actualizar(f"{m} | {a.rep} | semilla {s} | prueba grupo {g} ({k}/{len(pend)})")
        t0 = time.time()
        try:
            if m in SIN_ENTRENAR:
                if m not in cache_baselines:
                    cache_baselines[m] = SIN_ENTRENAR[m](np.arange(len(CAS)))
                sc = cache_baselines[m][te]
            else:
                sc = ENTRENADOS[m](tr, te, s)
            pd.DataFrame(dict(sesion=sesion[te], seg=seg[te], pareja=pareja[te], y=y[te], score=np.asarray(sc, dtype=float))).to_parquet(ruta_de(m, s, g), index=False)
            ap_ = u.ap_seguro(y[te], sc); prev = y[te].mean()
            p(f"  {m:11s} s{s} g{g}: PR-AUC {ap_:.3f} | prevalencia {prev:.3f} | {ap_ / prev:5.2f} x azar | {time.time() - t0:5.0f} s")
        except Exception as e:
            fallas.append((m, s, g)); p(f"  {m} s{s} g{g}: FALLO -> {type(e).__name__}: {e}"); log.lineas.append(traceback.format_exc())
        prog.actualizar(avance=1)
    prog.cerrar()

# ------------------------------------------------------------------------------------------------ resumen rapido
filas = []
for (m, s, g) in unidades:
    if os.path.exists(ruta_de(m, s, g)):
        d = pd.read_parquet(ruta_de(m, s, g)); prev = d["y"].mean(); ap_ = u.ap_seguro(d["y"], d["score"])
        filas.append(dict(modelo=m, rep=REP, semilla=s, grupo=g, n=len(d), positivos=int(d["y"].sum()), prevalencia=prev, pr_auc=ap_, x_azar=ap_ / prev))
R = pd.DataFrame(filas)
if len(R):
    R.to_csv(os.path.join(a.salida, f"tp_cuerpo_04_{REP}_{'-'.join(sorted(a.modelos))}_resultados.csv"), index=False)
    p("\nRESUMEN RAPIDO (promedio de las pruebas; la evaluacion estadistica completa es el script 05)")
    p(R.groupby("modelo").agg(pruebas=("grupo", "count"), pr_auc_prom=("pr_auc", "mean"), x_azar_prom=("x_azar", "mean"), x_azar_mediana=("x_azar", "median")).round(3).to_string())
if fallas:
    p("\nPRUEBAS QUE FALLARON:", fallas, "-> corrige y vuelve a ejecutar el mismo comando (solo repite las que faltan)")
log.guardar(os.path.join(a.salida, f"tp_cuerpo_04_{REP}_modelar.txt"), anexar=True)
