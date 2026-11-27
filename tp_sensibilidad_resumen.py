"""
tp_sensibilidad_resumen.py        (CRISP-DM, fase 5: EVALUACION; analisis de sensibilidad, canal CARA o CUERPO)
Junta en una sola tabla los resultados de la configuracion PRINCIPAL y de las variantes de sensibilidad, para ver si las
conclusiones dependen de decisiones propias (cobertura minima de datos validos, largo de la ventana, suavizado).

Cada variante se corrio en su propia carpeta de salida con los mismos scripts y se evaluo con el script 05 de esa carpeta
(mismas permutaciones, mismo bootstrap). Como las variantes de cobertura cambian QUE casos son usables, sus tablas no son
pareables caso a caso con la principal: se comparan las cifras (skill, IC 95 %, si supera el azar), no con una prueba pareada.
Lectura: si el skill de una variante cae dentro del IC 95 % de la principal y 'supera_azar' no cambia, la conclusion no depende
de esa decision. Si cambia, hay que decirlo en el informe.

Uso (desde la carpeta scripts_tp):
    python tp_sensibilidad_resumen.py --canal cara
    python tp_sensibilidad_resumen.py --canal cuerpo
Entradas: <carpeta>/tp_<canal>_05_resumen_modelos.csv de la principal y de cada variante que exista.
Salidas:  <principal>/tp_<canal>_sensibilidad.csv y .txt
"""
import os
import pandas as pd
import tp_util as u

ap = u.parser("Resumen del analisis de sensibilidad", root=False, anot=False)
ap.add_argument("--canal", choices=["cara", "cuerpo"], required=True)
ap.add_argument("--rep", default="seleccion", help="representacion que se compara (la principal)")
a, _ = ap.parse_known_args()
principal = u.SALIDA if a.canal == "cara" else u.SALIDA_CUERPO
pref = "salidas_tp_sens_" if a.canal == "cara" else "salidas_tp_cuerpo_sens_"
VARIANTES = [("principal (cobertura 70 % / 50 %, ventana 4 s, suavizado por defecto)", principal),
             ("cobertura laxa (50 % / 30 %)", pref + "cob_laxa"),
             ("cobertura estricta (90 % / 70 %)", pref + "cob_estricta"),
             ("ventana de 2 s (solo el segmento anotado)", pref + "vent2"),
             ("sin suavizado (Savitzky-Golay desactivado)", pref + "suav0")]
log = u.Log(); p = log.p
filas = []
for nombre, carpeta in VARIANTES:
    ruta = os.path.join(carpeta, f"tp_{a.canal}_05_resumen_modelos.csv")
    if not os.path.exists(ruta):
        p(f"(falta {ruta}: esa variante no se ha corrido o evaluado)"); continue
    T = pd.read_csv(ruta); T = T[T["rep"] == a.rep]
    for r in T.itertuples():
        filas.append(dict(variante=nombre, modelo=r.modelo, pr_auc=r.pr_auc_prom, prevalencia=r.prevalencia_prom, skill=r.skill_prom,
                          ic_inf=r.skill_ic_inf, ic_sup=r.skill_ic_sup, p_holm=r.p_holm, supera_azar=r.supera_azar, grupos=r.grupos))
R = pd.DataFrame(filas)
if len(R):
    R.round(4).to_csv(os.path.join(principal, f"tp_{a.canal}_sensibilidad.csv"), index=False)
    for m in ("energia", "sincronia", "rf", "minirocket", "transformer"):
        d = R[R["modelo"] == m]
        if len(d):
            p(f"\n=== {m} [{a.rep}] ==="); p(d.drop(columns="modelo").round(3).to_string(index=False))
    p("\nLectura: compara 'skill' y su IC entre variantes; los cambios de 'supera_azar' son los que importan para las conclusiones.")
    p("Cada variante tiene su propio conjunto de casos usables (cobertura) o de entrada (ventana, suavizado): no son pruebas pareadas.")
log.guardar(os.path.join(principal, f"tp_{a.canal}_sensibilidad.txt"))
