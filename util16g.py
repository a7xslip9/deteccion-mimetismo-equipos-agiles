"""
util16g.py
Funciones comunes de los scripts 16g_* (dataset completo: 16 grupos anotados, 32 sesiones).
No se ejecuta solo. Contiene: lectura de anotaciones, deteccion automatica de sesiones,
registro de resultados en texto y barra de progreso con porcentaje, tiempo transcurrido y tiempo restante.
"""
import argparse, os, re, shutil, sys, tempfile, threading, time
import pandas as pd

SALIDA = "salidas_16g"                 # carpeta de salida (distinta de 'salidas', que es la de los 2 grupos)
PAT_SES = re.compile(r"^\d{2}[AB]$")   # codigo de sesion: 01A, 15B, ...


def hms(s):
    if s is None or s != s:
        return "?"
    s = int(s); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parser(descripcion, root=True, anot=True):
    ap = argparse.ArgumentParser(description=descripcion)
    if root:
        ap.add_argument("--root", default="Root", help="carpeta con GrupoNN/A y GrupoNN/B (por defecto: Root)")
    if anot:
        ap.add_argument("--anot", default=os.path.join("Root", "mimicry_indicators.xlsx"))
    ap.add_argument("--salida", default=SALIDA, help=f"carpeta de salida (por defecto: {SALIDA})")
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
    """Sesiones (01A, 01B, ...) que tienen anotaciones en el Excel. Si se pasan grupos, solo esos."""
    ses = sorted({str(g)[:3] for g in ann["Grupo"].dropna() if PAT_SES.match(str(g)[:3])})
    if grupos:
        grupos = [g.zfill(2) for g in grupos]
        ses = [s for s in ses if s[:2] in grupos]
    return ses


class Log:
    """Guarda cada linea impresa para escribirla luego en un .txt."""
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
    """
    Barra de una linea:  [ 37.5%] detalle | transcurrido 04:12 | falta aprox. 07:10 | este paso 00:31
    El contador 'este paso' se actualiza cada segundo, aunque un paso sea largo: si ese numero sigue
    subiendo, el programa NO esta pegado, solo esta trabajando. El % avanza al terminar cada unidad de trabajo.
    """
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
