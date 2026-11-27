@echo off
REM EXTRA, terminal A (GPU): Transformer de cara con contexto corporal y Transformer de cuerpo sobre la etiqueta de cara.
REM Tiempo aproximado con tu GPU: cara_cuerpo ~1,5 a 2 h; cuerpo ~20 min.
python tp_cara_04_modelar.py --rep cara_cuerpo --modelos transformer
python tp_cara_04_modelar.py --rep cuerpo --modelos transformer
pause
