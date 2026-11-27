@echo off
REM EXTRA, terminal C (CPU): sensibilidad de la CARA (Random Forest y las dos bases, con la seleccion de puntos).
REM Tiempo aproximado: ~2 a 3 h en total (cada Random Forest de cara tarda ~20 min; el paso 02 sin suavizado es lo mas largo).
REM cobertura laxa
python tp_cara_03_construir_casos.py --series salidas_tp --salida salidas_tp_sens_cob_laxa --cob_actual 0.5 --cob_previo 0.3
python tp_cara_04_modelar.py --series salidas_tp --salida salidas_tp_sens_cob_laxa --rep seleccion --modelos energia sincronia rf
python tp_cara_05_evaluar.py --salida salidas_tp_sens_cob_laxa --rep_ref seleccion
REM cobertura estricta
python tp_cara_03_construir_casos.py --series salidas_tp --salida salidas_tp_sens_cob_estricta --cob_actual 0.9 --cob_previo 0.7
python tp_cara_04_modelar.py --series salidas_tp --salida salidas_tp_sens_cob_estricta --rep seleccion --modelos energia sincronia rf
python tp_cara_05_evaluar.py --salida salidas_tp_sens_cob_estricta --rep_ref seleccion
REM ventana de 2 s
python tp_cara_03_construir_casos.py --series salidas_tp --salida salidas_tp_sens_vent2
python tp_cara_04_modelar.py --series salidas_tp --salida salidas_tp_sens_vent2 --ventana 2 --rep seleccion --modelos energia sincronia rf
python tp_cara_05_evaluar.py --salida salidas_tp_sens_vent2 --rep_ref seleccion
REM sin suavizado (rehace las series de cara con --suavizado 0)
python tp_cara_02_preparar_series.py --suavizado 0 --salida salidas_tp_sens_suav0
python tp_cara_03_construir_casos.py --salida salidas_tp_sens_suav0
python tp_cara_04_modelar.py --salida salidas_tp_sens_suav0 --puntos salidas_tp\tp_cara_03b_puntos.json --rep seleccion --modelos energia sincronia rf
python tp_cara_05_evaluar.py --salida salidas_tp_sens_suav0 --rep_ref seleccion
pause
