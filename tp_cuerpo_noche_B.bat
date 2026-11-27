@echo off
REM Paso 2, terminal B (GPU) del cuerpo: Transformer.
python tp_cuerpo_04_modelar.py --rep seleccion --modelos transformer
python tp_cuerpo_04_modelar.py --rep todos --modelos transformer
python tp_cuerpo_04_modelar.py --rep descriptores --modelos transformer
pause
