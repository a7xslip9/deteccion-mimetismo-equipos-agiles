"""
tp_util.py   (serie "tp" = prueba con TODOS los puntos; canal CARA)
Funciones comunes a los scripts tp_cara_*. No se ejecuta solo.
Contiene: parametros comunes, lectura de anotaciones, deteccion de sesiones, registro en texto y barra de progreso.

Marco metodologico: los scripts siguen las fases de CRISP-DM (Wirth y Hipp, 2000; Chapman et al., 2000):
  01 comprension de los datos | 02 y 03 preparacion de los datos | 04 modelado | 05 evaluacion.
  (La fase 1, comprension del problema, y la 6, despliegue, estan descritas en LEEME_cara.md.)
"""
import argparse, os, re, shutil, sys, tempfile, threading, time
import numpy as np
import pandas as pd

SALIDA = "salidas_tp"                  # resultados de la cara
SALIDA_CUERPO = "salidas_tp_cuerpo"    # resultados del cuerpo (carpeta aparte)
PAT_SES = re.compile(r"^\d{2}[AB]$")      # codigo de sesion: 01A, 15B, ...

# ---------------------------------------------------------------------------------------------------------
# PARAMETROS COMUNES (cada uno esta justificado en DECISIONES_cara.md con su identificador D-xx)
# ---------------------------------------------------------------------------------------------------------
SEG_S = 2.0            # D-03: largo del segmento = unidad de anotacion del Excel (una fila cada 2 s)
N_CUADROS = 29         # D-04: cuadros por segmento tras remuestrear (~14,5 cuadros/s x 2 s)
HUECO_MAX = 5          # D-06: se interpolan huecos de hasta 5 cuadros (~0,35 s); mas largos quedan como faltantes
SG_VENTANA = 5         # D-07: filtro Savitzky-Golay, ventana (cuadros); 0 = sin suavizado
SG_ORDEN = 2           # D-07: orden del polinomio
COB_ACTUAL = 0.70      # D-05: minimo de cuadros validos en el segmento anotado, para cada persona
COB_PREVIO = 0.50      # D-05: minimo de cuadros validos en el segmento anterior (ventana de 4 s)
CLIP = 8.0             # D-09: valores estandarizados se recortan a +-8 (protege de fallos de la malla)
N_LM = 468             # puntos de la malla facial de MediaPipe (Kartynnik et al., 2019)

# Puntos "rigidos" de la cara (se mueven poco con la expresion): linea media de la frente y la nariz y
# las esquinas de los ojos. Sirven SOLO para alinear la cara cuadro a cuadro (D-08).
RIGIDOS = [10, 151, 168, 6, 197, 195, 5, 4, 1, 33, 133, 362, 263]
OJO_D_EXT, OJO_I_EXT = 33, 263                # esquinas externas de los ojos (escala = distancia entre ojos)
BOCA_I, BOCA_D, LABIO_SUP, LABIO_INF = 61, 291, 13, 14
PARP = {"od": (159, 145), "oi": (386, 374)}   # parpado superior / inferior de cada ojo
CEJA = {"cd": (105, 159), "ci": (334, 386)}   # ceja / parpado superior

NOMBRES_DESC = ["boca_apertura", "boca_ancho", "ojo_d", "ojo_i", "ceja_d", "ceja_i", "yaw", "pitch", "roll"]
NOMBRES_CAB = ["cab_tx", "cab_ty", "cab_logescala", "cab_giro"]
NOMBRES_TODOS = [f"lm{k}_{e}" for k in range(N_LM) for e in "xy"] + NOMBRES_CAB + ["yaw", "pitch", "roll"]
N_CANALES_TODOS = len(NOMBRES_TODOS)           # 936 + 4 + 3 = 943


# ---------------------------------------------------------------------------------------------------------
# CUERPO (scripts tp_cuerpo_*): pose de MediaPipe (33 puntos) y manos de MediaPipe (21 puntos por mano)
# ---------------------------------------------------------------------------------------------------------
N_POSE = 33
N_MANO = 21
UMBRAL_VIS = 0.5       # B-05: un punto de pose con visibilidad < 0,5 se considera no observado (convencion de MediaPipe)
HOMBRO_I, HOMBRO_D, CODO_I, CODO_D, MUNECA_I, MUNECA_D = 11, 12, 13, 14, 15, 16
NARIZ, BOCA_I_P, BOCA_D_P = 0, 9, 10
PUNTAS_MANO = [4, 8, 12, 16, 20]
PTS_MANO = list(range(1, N_MANO))              # la muneca (0) es el origen de la mano: no es un canal
NOMBRES_CUERPO_TODOS = ([f"p{k}_{e}" for k in range(N_POSE) for e in "xy"]
                        + [f"m{h}{k}_{e}" for h in "IR" for k in PTS_MANO for e in "xy"]
                        + [f"m{h}_pos_{e}" for h in "IR" for e in "xy"] + ["m_I_pres", "m_R_pres"]
                        + ["torso_logancho", "torso_inclinacion"])
N_CANALES_CUERPO_TODOS = len(NOMBRES_CUERPO_TODOS)    # 66 + 80 + 4 + 2 + 2 = 154
NOMBRES_DESC_CUERPO = ["torso_logancho", "torso_inclinacion", "torso_x", "torso_y", "codo_I", "codo_D", "muneca_alt_I", "muneca_alt_D",
                       "muneca_boca_I", "muneca_boca_D", "cabeza_dx", "cabeza_dy", "munecas_dist", "mano_pres_I", "mano_pres_D",
                       "mano_abierta_I", "mano_abierta_D"]


def hms(s):
    if s is None or s != s:
        return "?"
    s = int(s); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parser(descripcion, root=True, anot=True, salida=SALIDA):
    ap = argparse.ArgumentParser(description=descripcion)
    if root:
        ap.add_argument("--root", default="Root", help="carpeta con GrupoNN/A y GrupoNN/B (por defecto: Root)")
    if anot:
        ap.add_argument("--anot", default=os.path.join("Root", "mimicry_indicators.xlsx"))
    ap.add_argument("--salida", default=salida, help=f"carpeta de salida (por defecto: {salida})")
    ap.add_argument("--grupos", nargs="+", default=None,
                    help="solo para pruebas rapidas: limitar a estos grupos, por ejemplo --grupos 01 15")
    return ap


def leer_anotaciones(ruta):
    try:
        return pd.read_excel(ruta, dtype=str)
    except PermissionError:   # el archivo esta abierto en Excel
        tmp = os.path.join(tempfile.gettempdir(), "mimicry_copia.xlsx")
        shutil.copyfile(ruta, tmp)
        return pd.read_excel(tmp, dtype=str)


def sesiones_anotadas(ann, grupos=None):
    ses = sorted({str(g)[:3] for g in ann["Grupo"].dropna() if PAT_SES.match(str(g)[:3])})
    if grupos:
        grupos = [g.zfill(2) for g in grupos]
        ses = [s for s in ses if s[:2] in grupos]
    return ses


def ap_seguro(y, s):
    """PR-AUC (average precision) o NaN si no hay positivos."""
    from sklearn.metrics import average_precision_score
    y = np.asarray(y)
    return float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")


class Log:
    def __init__(self):
        self.lineas = []; self.prog = None

    def p(self, *x):
        s = " ".join(str(i) for i in x)
        self.lineas.append(s)
        if self.prog is not None:
            self.prog.mensaje(s)
        else:
            print(s)

    def guardar(self, ruta, anexar=False):
        with open(ruta, "a" if anexar else "w", encoding="utf-8") as f:
            f.write("\n".join(self.lineas) + "\n")
        print("\nGuardado:", ruta)


class Progreso:
    """Barra de una linea con porcentaje, tiempo transcurrido, tiempo estimado restante y contador del paso actual.
    El contador 'este paso' sube cada segundo: si sube, el programa NO esta pegado, solo trabajando."""
    def __init__(self, total, titulo, log=None):
        self.total = max(int(total), 1); self.hecho = 0.0
        self.t0 = time.time(); self.t_paso = self.t0; self.detalle = ""
        self.lock = threading.Lock(); self.stop = threading.Event(); self.ancho = 0
        self.log = log
        if log is not None:
            log.prog = self
        print(f"\n{titulo}")
        self.hilo = threading.Thread(target=self._latido, daemon=True); self.hilo.start()

    def _linea(self):
        el = time.time() - self.t0
        pct = 100.0 * self.hecho / self.total
        falta = el / self.hecho * (self.total - self.hecho) if self.hecho > 0 else None
        cols = shutil.get_terminal_size((120, 20)).columns
        fijo = f"[{pct:5.1f}%] "
        cola = f" | transcurrido {hms(el)} | falta aprox. {hms(falta)} | este paso {hms(time.time() - self.t_paso)}"
        libre = max(cols - len(fijo) - len(cola) - 2, 10)
        d = self.detalle if len(self.detalle) <= libre else self.detalle[:libre - 1] + "~"
        return fijo + d + cola

    def _pintar(self):
        s = self._linea()
        self.ancho = max(self.ancho, len(s))
        sys.stdout.write("\r" + s.ljust(self.ancho)); sys.stdout.flush()

    def _latido(self):
        while not self.stop.wait(1.0):
            with self.lock:
                self._pintar()

    def actualizar(self, detalle=None, avance=0.0):
        with self.lock:
            if detalle is not None:
                self.detalle = detalle
            if avance:
                self.hecho = min(self.hecho + avance, self.total); self.t_paso = time.time()
            self._pintar()

    def mensaje(self, texto):
        with self.lock:
            sys.stdout.write("\r" + " " * self.ancho + "\r" + texto + "\n"); self._pintar()

    def cerrar(self):
        self.stop.set()
        with self.lock:
            self.hecho = self.total; self._pintar()
            sys.stdout.write("\n"); sys.stdout.flush()
        if self.log is not None:
            self.log.prog = None


# ---------------------------------------------------------------------------------------------------------
# Acceso a los archivos de una sesion
# ---------------------------------------------------------------------------------------------------------
import glob, wave


def buscar(carpeta, sufijo, obligatorio=True):
    r = sorted(glob.glob(os.path.join(carpeta, "*" + sufijo)))
    if not r and obligatorio:
        raise FileNotFoundError(f"{sufijo} en {carpeta}")
    return r[0] if r else None


def carpeta_sesion(root, cod):
    return os.path.join(root, f"Grupo{cod[:2]}", cod[2])


def fps_sesion(carpeta):
    """Cuadros por segundo = cuadros de la sesion / duracion del audio (mismo criterio de toda la serie de scripts).
    Los cuadros se cuentan en poseTracker (un registro por persona y cuadro); si no existe, en faceTracker."""
    ruta = buscar(carpeta, ".poseTracker.parquet", obligatorio=False) or buscar(carpeta, ".faceTracker.parquet")
    n_fr = int(pd.read_parquet(ruta, columns=["frame"])["frame"].nunique())
    with wave.open(buscar(carpeta, ".wav")) as w:
        dur = w.getnframes() / w.getframerate()
    return n_fr, dur, n_fr / dur
