"""
16g_00_comprobar_datos.py
Revisa ANTES de correr nada que los datos de los 16 grupos esten completos en tu computador.
 - Detecta las sesiones anotadas en Root/mimicry_indicators.xlsx (deberian ser 32: 16 grupos x A y B).
 - Para cada sesion comprueba que existan en Root/GrupoNN/A (o B) los archivos que usan los scripts:
   poseTracker, faceTracker, faceDirection, handsTracker y el .wav.
 - Registra el hardware y las versiones de las librerias.
No modifica nada. Es rapido (menos de 1 minuto).

Uso (desde la carpeta scripts16g):
    python 16g_00_comprobar_datos.py
Salida: salidas_16g/16g_00_comprobar_datos.txt
"""
import glob, os, platform, sys, importlib
import util16g as u

a = u.parser("Comprobacion de datos de los 16 grupos").parse_args()
os.makedirs(a.salida, exist_ok=True)
log = u.Log(); p = log.p

p("=" * 70); p("ENTORNO"); p("=" * 70)
p("Sistema:", platform.platform(), "| nucleos logicos:", os.cpu_count(), "| Python:", sys.version.split()[0])
for lib in ["numpy", "pandas", "pyarrow", "openpyxl", "sklearn", "matplotlib", "scipy"]:
    try:
        p(f"{lib}: {importlib.import_module(lib).__version__}")
    except Exception:
        p(f"{lib}: NO INSTALADO")

p("\n" + "=" * 70); p("SESIONES ANOTADAS"); p("=" * 70)
ann = u.leer_anotaciones(a.anot)
ses = u.sesiones_anotadas(ann, a.grupos)
grupos = sorted({s[:2] for s in ses})
p(f"Sesiones con anotaciones: {len(ses)} | grupos: {len(grupos)}")
p("Grupos:", " ".join(grupos))
p("Filas de anotacion (segmentos de 2 s):", int(ann['Grupo'].str[:3].isin(ses).sum()))

SUFIJOS = {"pose": ".poseTracker.parquet", "cara": ".faceTracker.parquet", "direccion_cara": ".faceDirection.parquet",
           "manos": ".handsTracker.parquet", "audio": ".wav"}
OBLIG = {"pose", "cara", "manos", "audio"}      # faceDirection es opcional (sin ella no hay yaw/pitch/roll)
p("\n" + "=" * 70); p("ARCHIVOS POR SESION  (OK = esta; falta = no esta)"); p("=" * 70)
mal = []; avisos = []
for s in ses:
    carpeta = os.path.join(a.root, f"Grupo{s[:2]}", s[2])
    if not os.path.isdir(carpeta):
        p(f"{s}: NO EXISTE la carpeta {carpeta}"); mal.append(s); continue
    est = {k: bool(glob.glob(os.path.join(carpeta, "*" + suf))) for k, suf in SUFIJOS.items()}
    falta = [k for k, v in est.items() if not v and k in OBLIG]
    opc = [k for k, v in est.items() if not v and k not in OBLIG]
    p(f"{s}: " + " ".join(f"{k}={'OK' if v else 'falta'}" for k, v in est.items()))
    if falta: mal.append(s)
    if opc: avisos.append(s)

p("\n" + "=" * 70); p("RESUMEN"); p("=" * 70)
if not mal:
    p(f"TODO LISTO: las {len(ses)} sesiones tienen los archivos necesarios.")
else:
    p(f"PROBLEMAS en {len(mal)} sesiones: {' '.join(mal)}")
    p("Copia esas carpetas a Root/GrupoNN/A y Root/GrupoNN/B y vuelve a correr este script.")
if avisos:
    p(f"Aviso: sin faceDirection (yaw/pitch/roll) en {' '.join(avisos)}. Los scripts siguen, pero esas sesiones tendran menos descriptores de cara.")
log.guardar(os.path.join(a.salida, "16g_00_comprobar_datos.txt"))
