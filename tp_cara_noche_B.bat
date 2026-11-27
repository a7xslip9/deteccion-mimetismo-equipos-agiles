@echo off
REM Paso 2, terminal B (GPU): Transformer con puntos seleccionados, todos los puntos y descriptores.
python tp_cara_04_modelar.py --rep seleccion --modelos transformer
python tp_cara_04_modelar.py --rep todos --modelos transformer
python tp_cara_04_modelar.py --rep descriptores --modelos transformer
pause
