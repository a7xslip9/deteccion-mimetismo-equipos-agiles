@echo off
REM Paso 2, terminal A (CPU) del cuerpo.
python tp_cuerpo_04_modelar.py --rep seleccion --modelos energia sincronia rf
python tp_cuerpo_04_modelar.py --rep seleccion --modelos minirocket
python tp_cuerpo_04_modelar.py --rep todos --modelos energia sincronia rf
python tp_cuerpo_04_modelar.py --rep todos --modelos minirocket
python tp_cuerpo_04_modelar.py --rep descriptores --modelos energia sincronia rf minirocket
pause
