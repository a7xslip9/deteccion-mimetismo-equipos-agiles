"""
tp_cara_02_preparar_series.py        (CRISP-DM, fase 3: PREPARACION DE LOS DATOS, parte 1; canal CARA)
Convierte la malla facial de MediaPipe (468 puntos por persona y cuadro) en series limpias por persona y segmento de 2 s.
Genera DOS representaciones de la misma cara, para poder compararlas con evidencia (D-01):
  'todos'        los 468 puntos (x, y) alineados + 4 canales de movimiento de cabeza + yaw/pitch/roll = 943 canales.
  'descriptores' la representacion anterior: 6 medidas de forma (boca, ojos, cejas) + yaw/pitch/roll = 9 canales.

Pasos por persona (cada uno con su decision justificada en DECISIONES_cara.md):
  1. Ordena por cuadro y reindexa: los cuadros sin malla quedan como faltantes.                                  (D-05)
  2. Interpola linealmente huecos de hasta 5 cuadros; huecos mas largos quedan faltantes y NO se inventan.        (D-06)
  3. Suaviza cada coordenada en el tiempo con Savitzky-Golay (ventana 5, orden 2) para atenuar el redondeo a pixel.(D-07)
  4. Alinea la cara en cada cuadro (traslacion + escala + giro en el plano) a la forma de referencia de esa persona,
     con un analisis de Procrustes sobre 13 puntos rigidos (linea media y esquinas de ojos).                       (D-08)
     Lo que se quita (posicion, tamano, giro) se guarda como 4 canales de movimiento de cabeza.                    (D-11)
  5. Quita la forma neutra de cada persona (mediana de cada canal) y divide por UNA sola escala por persona; NO
     estandariza cada punto por separado, para no amplificar el ruido de los puntos casi inmoviles.               (D-10)
  6. Recorta a +-8 y agrupa en segmentos de 2 s, remuestreados a 29 cuadros.                                       (D-03, D-04, D-09)
  7. Registra por persona y segmento la fraccion de cuadros validos (cobertura), que usa el script 03.             (D-05)

Si el programa se corta, vuelve a ejecutarlo: se salta las sesiones ya listas (--rehacer fuerza todo).

Uso (desde la carpeta scripts_tp):
    python tp_cara_02_preparar_series.py
    python tp_cara_02_preparar_series.py --grupos 01 15           (prueba rapida)
    python tp_cara_02_preparar_series.py --suavizado 0            (sin suavizado, para medir su efecto)
Entradas: Root/GrupoNN/A|B/*.faceTracker.parquet, .faceDirection.parquet (opcional), .poseTracker.parquet o .wav
Salidas:  salidas_tp/tp_cara_02_<sesion>.npz (una por sesion), tp_cara_02_resumen.csv, tp_cara_02_canales.json,
          tp_cara_02_preparar_series.txt
Requiere: numpy, pandas, pyarrow, scipy, openpyxl
"""
import json, os, traceback, warnings
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
import tp_util as u

warnings.filterwarnings("ignore", category=RuntimeWarning)   # columnas enteras en NaN (p. ej. sin faceDirection) son esperables

ap = u.parser("Preparacion de series de cara (todos los puntos)")
ap.add_argument("--rehacer", action="store_true", help="volver a procesar aunque la sesion ya este lista")
ap.add_argument("--suavizado", type=int, default=u.SG_VENTANA, help="ventana Savitzky-Golay en cuadros (impar); 0 = sin suavizado")
a = ap.parse_args()
os.makedirs(a.salida, exist_ok=True)
log = u.Log(); p = log.p
RIG = u.RIGIDOS
IX_OJO_D, IX_OJO_I = RIG.index(u.OJO_D_EXT), RIG.index(u.OJO_I_EXT)
COLX = [f"{k}_x" for k in range(u.N_LM)]; COLY = [f"{k}_y" for k in range(u.N_LM)]


def interp_huecos(M):
    """M: T x D con NaN. Interpola solo huecos internos de hasta HUECO_MAX cuadros."""
    return pd.DataFrame(M).interpolate(method="linear", limit=u.HUECO_MAX, limit_area="inside").to_numpy()


def suavizar(L):
    """Savitzky-Golay en el tiempo sobre T x D (Savitzky y Golay, 1964). Los faltantes se rellenan solo para filtrar y se
    vuelven a marcar como faltantes despues."""
    if a.suavizado < 3:
        return L
    mask = np.isnan(L)
    rell = pd.DataFrame(L).ffill().bfill().to_numpy()
    if len(rell) < a.suavizado:
        return L
    out = savgol_filter(rell, window_length=a.suavizado, polyorder=u.SG_ORDEN, axis=0, mode="interp")
    out[mask] = np.nan
    return out


def ajustar_similitud(R, w):
    """Mejor transformacion de similitud (escala + giro + traslacion) de cada cuadro R (T x m, complejos) hacia la forma w (m).
    Solucion cerrada del Procrustes en 2D (Gower, 1975): w ~ a*R + b."""
    Rm = R.mean(1, keepdims=True); wm = w.mean()
    Rc = R - Rm; wc = w - wm
    a_ = (np.conj(Rc) * wc).sum(1) / (np.abs(Rc) ** 2).sum(1)
    b_ = wm - a_ * Rm[:, 0]
    return a_, b_


def referencia_procrustes(R):
    """Forma de referencia de la persona: Procrustes generalizado con 3 iteraciones sobre los cuadros validos (Gower, 1975).
    Se normaliza para que la distancia entre las esquinas externas de los ojos valga 1 (unidad = distancia entre ojos)."""
    ref = R[len(R) // 2]
    for _ in range(3):
        a_, b_ = ajustar_similitud(R, ref - ref.mean())
        ref = (a_[:, None] * R + b_[:, None]).mean(0)
        ref = (ref - ref.mean()) / np.abs(ref[IX_OJO_D] - ref[IX_OJO_I])
    return ref


def descriptores_forma(L):
    """6 medidas de forma por cuadro (L: T x 468 x 2), normalizadas por la distancia entre ojos del mismo cuadro.
    Son las mismas definiciones de la representacion anterior (para que la comparacion sea la misma cosa)."""
    def d(i, j):
        return np.hypot(L[:, i, 0] - L[:, j, 0], L[:, i, 1] - L[:, j, 1])
    esc = d(u.OJO_D_EXT, u.OJO_I_EXT)
    return np.stack([d(u.LABIO_SUP, u.LABIO_INF) / esc, d(u.BOCA_I, u.BOCA_D) / esc, d(*u.PARP["od"]) / esc,
                     d(*u.PARP["oi"]) / esc, d(*u.CEJA["cd"]) / esc, d(*u.CEJA["ci"]) / esc], axis=1)


def zscore_cols(M):
    mu = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0); sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (M - mu) / sd


def remuestrear(B, n=u.N_CUADROS):
    """B: cuadros x canales (sin NaN) -> n x canales por interpolacion lineal en el tiempo."""
    t = len(B)
    if t == n:
        return B
    pos = np.linspace(0, t - 1, n); i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, t - 1); w = (pos - i0)[:, None]
    return B[i0] * (1 - w) + B[i1] * w


def procesar_persona(Ft, Dt, frames, fps):
    """Ft: tabla de cara de una persona indexada por cuadro (reindexada). Dt: yaw/pitch/roll o None.
    Devuelve (X_todos, X_desc, seg, cobertura, info) por segmento."""
    T = len(frames)
    L = np.stack([Ft[COLX].to_numpy(float), Ft[COLY].to_numpy(float)], axis=2)           # T x 468 x 2
    falt0 = np.isnan(L[:, 0, 0]).mean()
    L = interp_huecos(L.reshape(T, -1)); falt1 = np.isnan(L[:, 0]).mean()
    L = suavizar(L).reshape(T, u.N_LM, 2)
    valid = ~np.isnan(L[:, 0, 0])
    if valid.sum() < 50:
        return None
    # ---- alineacion (D-08) ----
    Z = L[:, :, 0] + 1j * L[:, :, 1]
    R = Z[:, RIG]
    ref = referencia_procrustes(R[valid])
    a_, b_ = ajustar_similitud(R, ref)                                                     # NaN donde no hay cara
    Zal = a_[:, None] * Z + b_[:, None]
    resid = float(np.nanmedian(np.sqrt(np.nanmean(np.abs(a_[:, None] * R + b_[:, None] - ref) ** 2, axis=1))))
    X_lm = np.empty((T, 2 * u.N_LM)); X_lm[:, 0::2] = Zal.real; X_lm[:, 1::2] = Zal.imag
    # ---- movimiento de cabeza (D-11): posicion del centroide rigido, tamano relativo y giro en el plano ----
    cen = R.mean(1)
    eye_px = np.nanmedian(np.abs(R[:, IX_OJO_D] - R[:, IX_OJO_I]))
    cab = np.stack([(cen.real - np.nanmedian(cen.real)) / eye_px, (cen.imag - np.nanmedian(cen.imag)) / eye_px,
                    np.log(np.abs(a_)), np.angle(a_)], axis=1)
    # ---- yaw / pitch / roll de faceDirection ----
    ypr = np.full((T, 3), np.nan) if Dt is None else interp_huecos(Dt[["yaw", "pitch", "roll"]].to_numpy(float))
    # ---- normalizacion (D-10): forma neutra de la persona y UNA escala para los 936 canales de puntos ----
    med = np.nanmedian(X_lm, axis=0)
    C = X_lm - med
    s_p = float(np.sqrt(np.nanmean(C[valid] ** 2)))
    C = C / s_p
    X_todos = np.concatenate([C, zscore_cols(cab), zscore_cols(ypr)], axis=1)
    X_desc = np.concatenate([zscore_cols(descriptores_forma(L)), zscore_cols(ypr)], axis=1)
    X_todos = np.clip(X_todos, -u.CLIP, u.CLIP); X_desc = np.clip(X_desc, -u.CLIP, u.CLIP)
    seg_cuadro = np.floor((frames - 1) / fps / u.SEG_S).astype(int)
    xs, ds, ss, cs = [], [], [], []
    for s in np.unique(seg_cuadro):
        ix = np.where(seg_cuadro == s)[0]
        if len(ix) < 5:
            continue
        xs.append(remuestrear(np.nan_to_num(X_todos[ix], nan=0.0))); ds.append(remuestrear(np.nan_to_num(X_desc[ix], nan=0.0)))
        ss.append(int(s)); cs.append(float(valid[ix].mean()))
    info = dict(falt_inicial=falt0, falt_tras_interp=falt1, valido=float(valid.mean()), escala_persona=s_p, residuo_alineacion=resid)
    return np.stack(xs), np.stack(ds), np.array(ss), np.array(cs), info


def procesar(cod):
    c = u.carpeta_sesion(a.root, cod)
    n_fr, dur, fps = u.fps_sesion(c)
    F = pd.read_parquet(u.buscar(c, ".faceTracker.parquet"))
    rd = u.buscar(c, ".faceDirection.parquet", obligatorio=False)
    D = pd.read_parquet(rd) if rd else None
    frames = np.arange(1, n_fr + 1)
    X, Dd, tr, sg, cv, infos = [], [], [], [], [], []
    for t in sorted(F["track_id"].unique()):
        Ft = F[F["track_id"] == t].drop_duplicates("frame").set_index("frame").reindex(frames)
        Dt = None if D is None else D[D["track_id"] == t].drop_duplicates("frame").set_index("frame").reindex(frames)
        r = procesar_persona(Ft, Dt, frames, fps)
        if r is None:
            p(f"  {cod} persona {t}: menos de 50 cuadros con cara, se omite"); continue
        x, d, s, cov, info = r
        X.append(x); Dd.append(d); sg.append(s); cv.append(cov); tr.append(np.full(len(s), int(t), dtype=np.int8)); infos.append(info)
    np.savez_compressed(os.path.join(a.salida, f"tp_cara_02_{cod}.npz"),
                        X_todos=np.concatenate(X).astype(np.float16), X_desc=np.concatenate(Dd).astype(np.float32),
                        track=np.concatenate(tr), seg=np.concatenate(sg).astype(np.int32), cov=np.concatenate(cv).astype(np.float32),
                        fps=np.float32(fps), suavizado=np.int32(a.suavizado), con_direccion=np.int8(D is not None))
    I = pd.DataFrame(infos)
    p(f"  {cod}: cuadros={n_fr}, fps={fps:.2f}, personas={len(X)}, segmentos-persona={sum(len(s) for s in sg)} | "
      f"cuadros con cara tras limpiar: {I['valido'].mean() * 100:.1f} % (antes de interpolar: {(1 - I['falt_inicial'].mean()) * 100:.1f} %) | "
      f"residuo de alineacion: {I['residuo_alineacion'].median():.3f} (unid. distancia entre ojos) | faceDirection: {'si' if D is not None else 'NO'}")
    return dict(sesion=cod, fps=fps, personas=len(X), seg_persona=int(sum(len(s) for s in sg)), valido=I["valido"].mean(),
                residuo=I["residuo_alineacion"].median(), escala_persona_med=I["escala_persona"].median(), con_direccion=D is not None)


json.dump({"todos": u.NOMBRES_TODOS, "descriptores": u.NOMBRES_DESC}, open(os.path.join(a.salida, "tp_cara_02_canales.json"), "w"))
ann = u.leer_anotaciones(a.anot)
ses = u.sesiones_anotadas(ann, a.grupos)
pend = [s for s in ses if a.rehacer or not os.path.exists(os.path.join(a.salida, f"tp_cara_02_{s}.npz"))]
log.lineas.append(f"--- ejecucion: {len(ses)} sesiones, {len(pend)} por procesar, suavizado={a.suavizado} ---")
print(f"Sesiones anotadas: {len(ses)} | ya listas: {len(ses) - len(pend)} | por procesar: {len(pend)}")
filas, fallas = [], []
if pend:
    prog = u.Progreso(len(pend), "Preparando series de cara (una sesion = una unidad)", log)
    for k, cod in enumerate(pend, 1):
        prog.actualizar(f"sesion {cod} ({k}/{len(pend)})")
        try:
            filas.append(procesar(cod))
        except Exception as e:
            fallas.append(cod); p(f"  {cod}: FALLO -> {type(e).__name__}: {e}"); log.lineas.append(traceback.format_exc())
        prog.actualizar(avance=1)
    prog.cerrar()
if filas:
    ruta = os.path.join(a.salida, "tp_cara_02_resumen.csv")
    R = pd.DataFrame(filas)
    if os.path.exists(ruta):
        R = pd.concat([pd.read_csv(ruta), R]).drop_duplicates("sesion", keep="last")
    R.sort_values("sesion").to_csv(ruta, index=False)
listas = [s for s in ses if os.path.exists(os.path.join(a.salida, f"tp_cara_02_{s}.npz"))]
p(f"\nSesiones listas: {len(listas)} de {len(ses)}")
if fallas:
    p("Sesiones que fallaron:", " ".join(fallas), "(revisa con tp_cara_01 y vuelve a correr este script)")
log.guardar(os.path.join(a.salida, "tp_cara_02_preparar_series.txt"), anexar=True)
