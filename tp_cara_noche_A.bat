@echo off
REM Paso 2, terminal A (CPU): bases, Random Forest y MiniRocket con las 3 representaciones.
python tp_cara_04_modelar.py --rep seleccion --modelos energia sincronia rf
python tp_cara_04_modelar.py --rep seleccion --modelos minirocket
python tp_cara_04_modelar.py --rep todos --modelos energia sincronia rf
python tp_cara_04_modelar.py --rep todos --modelos minirocket
python tp_cara_04_modelar.py --rep descriptores --modelos energia sincronia rf minirocket
pause
