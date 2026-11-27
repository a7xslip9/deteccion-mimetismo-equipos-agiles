"""
tp_cuerpo_02_preparar_series.py        (CRISP-DM, fase 3: PREPARACION DE LOS DATOS, parte 1; canal CUERPO)
Convierte la pose de MediaPipe (33 puntos) y las manos de MediaPipe (21 puntos por mano) en series limpias por persona y
segmento de 2 s. Genera DOS representaciones del mismo cuerpo, para compararlas con evidencia (B-02):
  'todos'        154 canales: 33 puntos de pose (x, y) + 20 puntos por mano (x, y, relativos a la muneca) + posicion de cada
                 muneca de mano + presencia de cada mano + 2 canales de torso (ancho e inclinacion de hombros).
  'descriptores' 17 medidas interpretables (torso, codos, altura de munecas, distancia mano-boca, mano abierta, ...).
El script 03b define una tercera, 'seleccion', como subconjunto de 'todos' (sin recalcular nada).

Pasos por persona (cada decision con su justificacion en DECISIONES_cuerpo.md):
  1. Reindexa por cuadro. Un punto de pose con visibilidad < 0,5 se toma como NO observado (faltante).             (B-05)
     Filas de mano duplicadas (mismo cuadro, persona y mano): se conserva la primera.                              (B-06)
  2. Interpola huecos de hasta 5 cuadros (~0,35 s); mas largos quedan faltantes y NO se inventan.                  (D-06)
  3. Suaviza cada coordenada en el tiempo con Savitzky-Golay (ventana 5, orden 2).                                (D-07)
  4. Pose: se expresa en el marco del cuerpo de la persona: origen = punto medio de los hombros (mediana de la
     sesion) y escala = ancho de hombros (mediana de la sesion). Son CONSTANTES por persona y sesion, NO se
     normaliza cuadro a cuadro, porque eso borraria inclinarse hacia atras/adelante, que es una de las acciones
     anotadas (lean backwards).                                                                                    (B-03)
  5. Manos: cada punto relativo a la muneca de esa mano y dividido por el tamano de mano de la persona (mediana de la
     distancia muneca-base del dedo medio). La muneca ademas se guarda en el marco del cuerpo. Si la mano no esta
     detectada: canales en 0 y presencia = 0.                                                                      (B-04)
  6. NO se estandariza cada canal por su desviacion y NO se resta la postura mediana: las posturas absolutas (mano en
     la boca, brazos cruzados) son lo que se compara entre dos personas.                                            (B-03)
  7. Recorta a +-8, agrupa en segmentos de 2 s remuestreados a 29 cuadros y registra la cobertura (cuadros con ambos
     hombros visibles) que usa el script 03.                                                                       (D-03, D-04, D-05, D-09)

Si el programa se corta, vuelve a ejecutarlo: se salta las sesiones ya listas (--rehacer fuerza todo).

Uso (desde la carpeta scripts_tp):
    python tp_cuerpo_02_preparar_series.py
    python tp_cuerpo_02_preparar_series.py --grupos 01 15           (prueba rapida)
    python tp_cuerpo_02_preparar_series.py --suavizado 0            (sin suavizado, para medir su efecto)
Entradas: Root/GrupoNN/A|B/*.poseTracker.parquet, *.handsTracker.parquet, *.wav (para los cuadros por segundo)
Salidas:  salidas_tp_cuerpo/tp_cuerpo_02_<sesion>.npz, tp_cuerpo_02_resumen.csv, tp_cuerpo_02_canales.json, tp_cuerpo_02_preparar_series.txt
"""
import json, os, traceback, warnings
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
import tp_util as u

warnings.filterwarnings("ignore", category=RuntimeWarning)

ap = u.parser("Preparacion de series de cuerpo", salida=u.SALIDA_CUERPO)
ap.add_argument("--rehacer", action="store_true", help="volver a procesar aunque la sesion ya este lista")
ap.add_argument("--suavizado", type=int, default=u.SG_VENTANA, help="ventana Savitzky-Golay en cuadros (impar); 0 = sin suavizado")
a = ap.parse_args()
os.makedirs(a.salida, exist_ok=True)
log = u.Log(); p = log.p
NP, NM = u.N_POSE, u.N_MANO
NOM_HAND = {"I": "Left", "R": "Right"}      # B-06: 'Left'/'Right' tal como viene en el dataset; solo se usa para distinguir las dos manos


def interp_huecos(M):
    return pd.DataFrame(M).interpolate(method="linear", limit=u.HUECO_MAX, limit_area="inside").to_numpy()


def suavizar(L):
    """Savitzky-Golay en el tiempo (Savitzky y Golay, 1964) sobre T x D; los faltantes se rellenan solo para filtrar y vuelven a ser faltantes."""
    if a.suavizado < 3 or len(L) < a.suavizado:
        return L
    mask = np.isnan(L)
    rell = pd.DataFrame(L).ffill().bfill().fillna(0).to_numpy()
    out = savgol_filter(rell, window_length=a.suavizado, polyorder=u.SG_ORDEN, axis=0, mode="interp")
    out[mask] = np.nan
    return out


def remuestrear(B, n=u.N_CUADROS):
    t = len(B)
    if t == n:
        return B
    pos = np.linspace(0, t - 1, n); i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, t - 1); w = (pos - i0)[:, None]
    return B[i0] * (1 - w) + B[i1] * w


def zscore_cols(M):
    mu = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0); sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (M - mu) / sd


def angulo(a_, b_, c_):
    """Angulo en b entre los segmentos b->a y b->c (radianes); a_, b_, c_: T x 2."""
    v1, v2 = a_ - b_, c_ - b_
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.arccos(np.clip(cos, -1, 1))


def leer_mano(Ht, frames, h):
    """T x 21 x 2 de la mano h ('Left'/'Right') de una persona, NaN donde no se detecto. Duplicados: se conserva la primera fila (B-06)."""
    T = len(frames)
    if Ht is None:
        return np.full((T, NM, 2), np.nan)
    d = Ht[Ht["hand"] == h].drop_duplicates("frame").set_index("frame").reindex(frames)
    return np.stack([np.stack([d[f"{k}_x"].to_numpy(float), d[f"{k}_y"].to_numpy(float)], 1) for k in range(NM)], 1)


def procesar_persona(Pt, Ht, frames, fps):
    T = len(frames)
    L = np.full((T, NP, 2), np.nan); VIS = np.zeros((T, NP))
    for k in range(NP):
        x, y, v = (Pt[f"{k}_{e}"].to_numpy(float) for e in ("x", "y", "v"))
        vis = np.nan_to_num(v, nan=0.0)
        ok = vis >= u.UMBRAL_VIS
        L[ok, k, 0] = x[ok]; L[ok, k, 1] = y[ok]; VIS[:, k] = vis
    vis_pose = (VIS >= u.UMBRAL_VIS).mean(0)                                    # fraccion de cuadros con el punto visible (antes de interpolar)
    L = suavizar(interp_huecos(L.reshape(T, -1))).reshape(T, NP, 2)
    valid = ~np.isnan(L[:, u.HOMBRO_I, 0]) & ~np.isnan(L[:, u.HOMBRO_D, 0])
    if valid.sum() < 50:
        return None
    mid = (L[:, u.HOMBRO_I] + L[:, u.HOMBRO_D]) / 2
    ancho = np.linalg.norm(L[:, u.HOMBRO_I] - L[:, u.HOMBRO_D], axis=1)
    o = np.nanmedian(mid[valid], axis=0); s = float(np.nanmedian(ancho[valid]))        # B-03: origen y escala CONSTANTES de la persona
    Pc = ((L - o) / s).reshape(T, -1)
    # ---- manos (B-04) ----
    Hc, pos, pres, hand_d = [], [], [], {}
    hs_list = []
    manos = {h: leer_mano(Ht, frames, n) for h, n in NOM_HAND.items()}
    for h, Mh in manos.items():
        d = np.linalg.norm(Mh[:, 9] - Mh[:, 0], axis=1)
        if np.isfinite(d).any():
            hs_list.append(d[np.isfinite(d)])
    hs = float(np.median(np.concatenate(hs_list))) if hs_list else np.nan           # tamano de mano de la persona (constante)
    for h, Mh in manos.items():
        M = suavizar(interp_huecos(Mh.reshape(T, -1))).reshape(T, NM, 2)
        pr = ~np.isnan(M[:, 0, 0])
        rel = (M[:, 1:] - M[:, :1]) / hs if np.isfinite(hs) else np.full((T, NM - 1, 2), np.nan)
        Hc.append(rel.reshape(T, -1))
        pos.append((M[:, 0] - o) / s); pres.append(pr.astype(float)); hand_d[h] = (M, rel, pr)
    # ---- torso ----
    torso = np.stack([np.log(np.maximum(ancho, 1e-6) / s), np.arctan2(L[:, u.HOMBRO_D, 1] - L[:, u.HOMBRO_I, 1], L[:, u.HOMBRO_D, 0] - L[:, u.HOMBRO_I, 0])], 1)
    X_todos = np.concatenate([Pc, Hc[0], Hc[1], pos[0], pos[1], np.stack(pres, 1), torso], axis=1)
    assert X_todos.shape[1] == u.N_CANALES_CUERPO_TODOS, X_todos.shape
    # ---- descriptores (B-08) ----
    Lm = L
    muneca_i, muneca_d = Lm[:, u.MUNECA_I], Lm[:, u.MUNECA_D]
    boca = (Lm[:, u.BOCA_I_P] + Lm[:, u.BOCA_D_P]) / 2
    abierta = []
    for h in ("I", "R"):
        M, rel, pr = hand_d[h]
        ab = np.linalg.norm(M[:, u.PUNTAS_MANO] - M[:, :1], axis=2).mean(1) / hs if np.isfinite(hs) else np.full(T, np.nan)
        abierta.append(np.where(pr, ab, np.nan))
    D = np.stack([torso[:, 0], torso[:, 1], (mid[:, 0] - o[0]) / s, (mid[:, 1] - o[1]) / s,
                  angulo(Lm[:, u.HOMBRO_I], Lm[:, u.CODO_I], muneca_i), angulo(Lm[:, u.HOMBRO_D], Lm[:, u.CODO_D], muneca_d),
                  (muneca_i[:, 1] - mid[:, 1]) / s, (muneca_d[:, 1] - mid[:, 1]) / s,
                  np.linalg.norm(muneca_i - boca, axis=1) / s, np.linalg.norm(muneca_d - boca, axis=1) / s,
                  (Lm[:, u.NARIZ, 0] - mid[:, 0]) / s, (Lm[:, u.NARIZ, 1] - mid[:, 1]) / s,
                  np.linalg.norm(muneca_i - muneca_d, axis=1) / s, pres[0], pres[1], abierta[0], abierta[1]], axis=1)
    assert D.shape[1] == len(u.NOMBRES_DESC_CUERPO)
    X_desc = zscore_cols(D)
    X_todos = np.clip(np.nan_to_num(X_todos, nan=0.0), -u.CLIP, u.CLIP); X_desc = np.clip(np.nan_to_num(X_desc, nan=0.0), -u.CLIP, u.CLIP)
    seg_cuadro = np.floor((frames - 1) / fps / u.SEG_S).astype(int)
    xs, ds, ss, cs = [], [], [], []
    for sg in np.unique(seg_cuadro):
        ix = np.where(seg_cuadro == sg)[0]
        if len(ix) < 5:
            continue
        xs.append(remuestrear(X_todos[ix])); ds.append(remuestrear(X_desc[ix])); ss.append(int(sg)); cs.append(float(valid[ix].mean()))
    info = dict(valido=float(valid.mean()), escala_hombros_px=s, tam_mano_px=hs, mano_I=float(pres[0].mean()), mano_D=float(pres[1].mean()))
    return np.stack(xs), np.stack(ds), np.array(ss), np.array(cs), vis_pose, info


def procesar(cod):
    c = u.carpeta_sesion(a.root, cod)
    n_fr, dur, fps = u.fps_sesion(c)
    P = pd.read_parquet(u.buscar(c, ".poseTracker.parquet"))
    rh = u.buscar(c, ".handsTracker.parquet", obligatorio=False)
    H = pd.read_parquet(rh) if rh else None
    if H is None:
        p(f"  {cod}: AVISO: sin archivo de manos; los canales de manos quedan en 0")
    frames = np.arange(1, n_fr + 1)
    X, Dd, tr, sg, cv, vp, infos = [], [], [], [], [], [], []
    for t in sorted(P["track_id"].unique()):
        Pt = P[P["track_id"] == t].drop_duplicates("frame").set_index("frame").reindex(frames)
        Ht = None if H is None else H[H["track_id"] == t]
        r = procesar_persona(Pt, Ht, frames, fps)
        if r is None:
            p(f"  {cod} persona {t}: menos de 50 cuadros con hombros visibles, se omite"); continue
        x, d, s, cov, vis, info = r
        X.append(x); Dd.append(d); sg.append(s); cv.append(cov); vp.append(vis); tr.append(np.full(len(s), int(t), dtype=np.int8)); infos.append(info)
    np.savez_compressed(os.path.join(a.salida, f"tp_cuerpo_02_{cod}.npz"),
                        X_todos=np.concatenate(X).astype(np.float16), X_desc=np.concatenate(Dd).astype(np.float32),
                        track=np.concatenate(tr), seg=np.concatenate(sg).astype(np.int32), cov=np.concatenate(cv).astype(np.float32),
                        vis_pose=np.stack(vp).astype(np.float32), fps=np.float32(fps), suavizado=np.int32(a.suavizado))
    I = pd.DataFrame(infos)
    p(f"  {cod}: cuadros={n_fr}, fps={fps:.2f}, personas={len(X)}, segmentos-persona={sum(len(s) for s in sg)} | "
      f"hombros visibles: {I['valido'].mean() * 100:.1f} % | ancho de hombros: {I['escala_hombros_px'].median():.0f} px | "
      f"mano I/D detectada: {I['mano_I'].mean() * 100:.0f} % / {I['mano_D'].mean() * 100:.0f} %")
    return dict(sesion=cod, fps=fps, personas=len(X), seg_persona=int(sum(len(s) for s in sg)), valido=I["valido"].mean(),
                escala_hombros_px=I["escala_hombros_px"].median(), mano_I=I["mano_I"].mean(), mano_D=I["mano_D"].mean())


json.dump({"todos": u.NOMBRES_CUERPO_TODOS, "descriptores": u.NOMBRES_DESC_CUERPO}, open(os.path.join(a.salida, "tp_cuerpo_02_canales.json"), "w"))
ann = u.leer_anotaciones(a.anot)
ses = u.sesiones_anotadas(ann, a.grupos)
pend = [s for s in ses if a.rehacer or not os.path.exists(os.path.join(a.salida, f"tp_cuerpo_02_{s}.npz"))]
log.lineas.append(f"--- ejecucion: {len(ses)} sesiones, {len(pend)} por procesar, suavizado={a.suavizado} ---")
print(f"Sesiones anotadas: {len(ses)} | ya listas: {len(ses) - len(pend)} | por procesar: {len(pend)}")
filas, fallas = [], []
if pend:
    prog = u.Progreso(len(pend), "Preparando series de cuerpo (una sesion = una unidad)", log)
    for k, cod in enumerate(pend, 1):
        prog.actualizar(f"sesion {cod} ({k}/{len(pend)})")
        try:
            filas.append(procesar(cod))
        except Exception as e:
            fallas.append(cod); p(f"  {cod}: FALLO -> {type(e).__name__}: {e}"); log.lineas.append(traceback.format_exc())
        prog.actualizar(avance=1)
    prog.cerrar()
if filas:
    ruta = os.path.join(a.salida, "tp_cuerpo_02_resumen.csv")
    R = pd.DataFrame(filas)
    if os.path.exists(ruta):
        R = pd.concat([pd.read_csv(ruta), R]).drop_duplicates("sesion", keep="last")
    R.sort_values("sesion").to_csv(ruta, index=False)
listas = [s for s in ses if os.path.exists(os.path.join(a.salida, f"tp_cuerpo_02_{s}.npz"))]
p(f"\nSesiones listas: {len(listas)} de {len(ses)}")
if fallas:
    p("Sesiones que fallaron:", " ".join(fallas), "(revisa con tp_cuerpo_01 y vuelve a correr este script)")
log.guardar(os.path.join(a.salida, "tp_cuerpo_02_preparar_series.txt"), anexar=True)
