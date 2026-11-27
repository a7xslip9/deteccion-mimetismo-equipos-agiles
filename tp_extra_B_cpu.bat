@echo off
REM EXTRA, terminal B (CPU): (1) cara con contexto corporal y cuerpo solo, sobre la etiqueta de cara; (2) sensibilidad del CUERPO.
REM Tiempo aproximado: parte 1 ~40 min; parte 2 ~40 min.
REM (1) cara + cuerpo y solo cuerpo, etiqueta de cara
python tp_cara_04_modelar.py --rep cara_cuerpo --modelos energia sincronia rf
python tp_cara_04_modelar.py --rep cuerpo --modelos energia sincronia rf
REM (2) sensibilidad del cuerpo: cobertura laxa
python tp_cuerpo_03_construir_casos.py --series salidas_tp_cuerpo --salida salidas_tp_cuerpo_sens_cob_laxa --cob_actual 0.5 --cob_previo 0.3
python tp_cuerpo_04_modelar.py --series salidas_tp_cuerpo --salida salidas_tp_cuerpo_sens_cob_laxa --rep seleccion --modelos energia sincronia rf
python tp_cuerpo_05_evaluar.py --salida salidas_tp_cuerpo_sens_cob_laxa --rep_ref seleccion
REM cobertura estricta
python tp_cuerpo_03_construir_casos.py --series salidas_tp_cuerpo --salida salidas_tp_cuerpo_sens_cob_estricta --cob_actual 0.9 --cob_previo 0.7
python tp_cuerpo_04_modelar.py --series salidas_tp_cuerpo --salida salidas_tp_cuerpo_sens_cob_estricta --rep seleccion --modelos energia sincronia rf
python tp_cuerpo_05_evaluar.py --salida salidas_tp_cuerpo_sens_cob_estricta --rep_ref seleccion
REM ventana de 2 s
python tp_cuerpo_03_construir_casos.py --series salidas_tp_cuerpo --salida salidas_tp_cuerpo_sens_vent2
python tp_cuerpo_04_modelar.py --series salidas_tp_cuerpo --salida salidas_tp_cuerpo_sens_vent2 --ventana 2 --rep seleccion --modelos energia sincronia rf
python tp_cuerpo_05_evaluar.py --salida salidas_tp_cuerpo_sens_vent2 --rep_ref seleccion
REM sin suavizado (rehace las series del cuerpo con --suavizado 0)
python tp_cuerpo_02_preparar_series.py --suavizado 0 --salida salidas_tp_cuerpo_sens_suav0
python tp_cuerpo_03_construir_casos.py --salida salidas_tp_cuerpo_sens_suav0
python tp_cuerpo_04_modelar.py --salida salidas_tp_cuerpo_sens_suav0 --puntos salidas_tp_cuerpo\tp_cuerpo_03b_puntos.json --rep seleccion --modelos energia sincronia rf
python tp_cuerpo_05_evaluar.py --salida salidas_tp_cuerpo_sens_suav0 --rep_ref seleccion
pause
