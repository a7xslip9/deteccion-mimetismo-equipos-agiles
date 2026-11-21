"""
16g_03_medidas_sincronia.py   (MS2: medidas de sincronia; equivale al script 05)
Para cada caso usable calcula, sobre un contexto de 6 s (2 s antes, el segmento de 2 s, 2 s despues):
  xc_pos_max, xc_pos_lag, xc_vel_max, xc_vel_lag (correlacion cruzada), dtw_pos, dtw_vel (DTW con banda de +-2 s),
  dur_sim (tramo mas largo con similitud de velocidad) y mov_min, mov_max (movimiento de cada persona).
Mismas definiciones que el script 05.

ES EL SCRIPT MAS LENTO DE LA PARTE DE MEDIDAS (el DTW). La barra muestra el % y el tiempo restante.
Guarda cada sesion al terminarla: si se corta, vuelve a ejecutarlo y continua donde quedo (--rehacer para forzar todo).

Uso (desde la carpeta scripts16g):   python 16g_03_medidas_sincronia.py     (o con --grupos 01 15 para probar)
Entradas: salidas_16g/16g_01_series_<cod>.parquet, salidas_16g/16g_02_casos.parquet
Salidas:  salidas_16g/16g_03_car_<cod>.parquet (una por sesion), 16g_03_caracteristicas.parquet, 16g_03_medidas_sincronia.txt
"""
import os
import numpy as np, pandas as pd
import util16g as u

W_ANTES, W_SEG, W_DESP = 2.0, 2.0, 2.0; LAG_S = 2.0; BANDA_S = 2.0; COS_MIN = 0.5; CHUNK = 100

ap = u.parser("MS2: medidas de sincronia (16 grupos)", root=False, anot=False)
ap.add_argument("--rehacer", action="store_true")
a = ap.parse_args()
log = u.Log(); p = log.p

def zfill_ventana(X):
    m = np.nanmean(np.where(np.isnan(X), np.nan, X), axis=1, keepdims=True)
    m = np.where(np.isnan(m), 0.0, m)
    return np.where(np.isnan(X), m, X)

def zscore(X, eps=1e-6):
    mu = X.mean(axis=1, keepdims=True); sd = X.std(axis=1, keepdims=True)
    Z = (X - mu) / np.where(sd < eps, 1.0, sd)
    return np.where(sd < eps, 0.0, Z)

def xcorr(A, B, m):
    N, T, K = A.shape
    za, zb = zscore(A), zscore(B)
    res = np.empty((N, 2 * m + 1, K))
    for idx, lag in enumerate(range(-m, m + 1)):
        if lag >= 0: x, y = za[:, :T - lag], zb[:, lag:]
        else:        x, y = za[:, -lag:], zb[:, :T + lag]
        res[:, idx] = (x * y).mean(axis=1)
    return res

def dtw(A, B, w):
    N, T, K = A.shape
    aa = (A ** 2).sum(-1)[:, :, None]; bb = (B ** 2).sum(-1)[:, None, :]
    cost = np.sqrt(np.maximum(aa + bb - 2 * np.einsum("nik,njk->nij", A, B), 0))
    acc = np.full((N, T + 1, T + 1), np.inf); acc[:, 0, 0] = 0
    for i in range(1, T + 1):
        for j in range(max(1, i - w), min(T, i + w) + 1):
            acc[:, i, j] = cost[:, i - 1, j - 1] + np.minimum(np.minimum(acc[:, i - 1, j], acc[:, i, j - 1]), acc[:, i - 1, j - 1])
    return acc[:, T, T] / T

def racha_max(v):
    mx = run = 0
    for e in v:
        run = run + 1 if e else 0; mx = max(mx, run)
    return mx

def caracteristicas(A, B, fps, avance):
    """avance(n): avisa que se terminaron n 'unidades' (3 por caso: DTW pos, DTW vel, duracion)."""
    N, T, K = A.shape
    m = int(round(LAG_S * fps)); w = int(round(BANDA_S * fps))
    f = {}
    for nombre, (X, Y) in {"pos": (A, B), "vel": (np.diff(A, axis=1), np.diff(B, axis=1))}.items():
        xc = xcorr(X, Y, m)
        mx = xc.max(axis=1); lag = xc.argmax(axis=1) - m
        f[f"xc_{nombre}_max"] = mx.mean(axis=1)
        f[f"xc_{nombre}_lag"] = np.abs(np.median(lag, axis=1)) / fps
        if nombre == "vel": lag_signed = np.median(lag, axis=1).astype(int)
        dd = np.empty(N)
        for k in range(0, N, CHUNK):
            dd[k:k + CHUNK] = dtw(X[k:k + CHUNK], Y[k:k + CHUNK], w); avance(len(X[k:k + CHUNK]))
        f[f"dtw_{nombre}"] = dd
    VA, VB = np.diff(A, axis=1), np.diff(B, axis=1)
    dur = np.zeros(N)
    for n in range(N):
        l = lag_signed[n]
        x, y = (VA[n, :VA.shape[1] - l], VB[n, l:]) if l >= 0 else (VA[n, -l:], VB[n, :VB.shape[1] + l])
        num = (x * y).sum(1); den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        cos = np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1), 0.0)
        dur[n] = racha_max(cos > COS_MIN) / fps
        if n % 200 == 199: avance(200)
    avance(N % 200)
    f["dur_sim"] = dur
    ma = np.linalg.norm(VA, axis=2).mean(axis=1) * fps; mb = np.linalg.norm(VB, axis=2).mean(axis=1) * fps
    f["mov_min"] = np.minimum(ma, mb); f["mov_max"] = np.maximum(ma, mb)
    return pd.DataFrame(f)

casos = pd.read_parquet(os.path.join(a.salida, "16g_02_casos.parquet"))
casos = casos[casos["usable"]]
ses = sorted(casos["sesion"].unique())
ruta_s = lambda c: os.path.join(a.salida, f"16g_03_car_{c}.parquet")
pend = [s for s in ses if a.rehacer or not os.path.exists(ruta_s(s))]
print(f"Sesiones: {len(ses)} | ya listas: {len(ses) - len(pend)} | por procesar: {len(pend)}")
total = int(sum((casos["sesion"] == s).sum() for s in pend)) * 3
if pend:
    prog = u.Progreso(total, "MS2: medidas de sincronia (el % cuenta casos; las sesiones ya hechas no se repiten)", log)
    for k, cod in enumerate(pend, 1):
        S = pd.read_parquet(os.path.join(a.salida, f"16g_01_series_{cod}.parquet"))
        tt = S[S["track_id"] == 1]["t"].values
        fps = 1.0 / np.median(np.diff(tt)); n = len(tt)
        L = int(round((W_ANTES + W_SEG + W_DESP) * fps))
        cols = {"cuerpo": [f"b{i}_{e}" for i in range(13, 23) for e in "xy"], "cara": [c for c in S.columns if c.startswith("c_")]}
        mats = {}
        for canal, cl in cols.items():
            for t_ in (1, 2, 3, 4):
                M = S[S["track_id"] == t_].sort_values("frame")[cl].values.astype(float)
                if canal == "cara":
                    M = (M - np.nanmean(M, axis=0)) / np.where(np.nanstd(M, axis=0) < 1e-9, 1, np.nanstd(M, axis=0))
                mats[(canal, t_)] = M
        partes = []
        for canal in ("cuerpo", "cara"):
            prog.actualizar(f"sesion {cod} ({k}/{len(pend)}) canal {canal}")
            C = casos[(casos["sesion"] == cod) & (casos["canal"] == canal)].copy()
            start = np.round((2 * C["seg"].values - W_ANTES) * fps).astype(int)
            valid = (start >= 0) & (start + L <= n)
            prog.actualizar(avance=3 * int((~valid).sum()))   # casos de borde: no se calculan
            C = C[valid]; start = start[valid]
            i_ = C["pareja"].str[0].astype(int).values; j_ = C["pareja"].str[2].astype(int).values
            idx = start[:, None] + np.arange(L)[None, :]
            A = np.stack([mats[(canal, i)][ix] for i, ix in zip(i_, idx)])
            B = np.stack([mats[(canal, j)][ix] for j, ix in zip(j_, idx)])
            A, B = zfill_ventana(A), zfill_ventana(B)
            F = caracteristicas(A, B, fps, lambda n_: prog.actualizar(avance=n_))
            F.index = C.index
            partes.append(pd.concat([C[["sesion", "grupo", "seg", "pareja", "canal", "y", "y_sync", "y_async"]], F], axis=1))
        pd.concat(partes, ignore_index=True).to_parquet(ruta_s(cod), index=False)
    prog.cerrar()

R = pd.concat([pd.read_parquet(ruta_s(s)) for s in ses if os.path.exists(ruta_s(s))], ignore_index=True)
R.to_parquet(os.path.join(a.salida, "16g_03_caracteristicas.parquet"), index=False)
feats = [c for c in R.columns if c.startswith(("xc_", "dtw_", "dur_", "mov_"))]
p("sesiones con caracteristicas:", R["sesion"].nunique(), "| casos:", len(R), "| NaN en caracteristicas:", int(R[feats].isna().sum().sum()))
p("\nMediana de cada caracteristica segun la etiqueta (0 = sin mimetismo, 1 = con mimetismo):")
for canal, d in R.groupby("canal"):
    p(f"[{canal}]  positivos={int(d['y'].sum())}, negativos={int((d['y'] == 0).sum())}")
    p(d.groupby("y")[feats].median().round(4).T.to_string())
log.guardar(os.path.join(a.salida, "16g_03_medidas_sincronia.txt"))
