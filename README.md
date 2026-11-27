# Detección de mimetismo corporal y facial en equipos ágiles

Código de la etapa actual del trabajo de título (Universidad de Valparaíso, Ingeniería Civil Informática). Detecta mimetismo entre pares de personas a partir de las series de puntos clave del conjunto de datos multimodal de equipos ágiles, y lo evalúa contra las anotaciones humanas dejando un grupo fuera. Compara tres modelos: Random Forest (referencia), MiniRocket y Transformer, además de dos bases sin aprendizaje (energía y sincronía).

Las decisiones de cada script están en `DECISIONES_cara.md` (D-xx) y `DECISIONES_cuerpo.md` (B-xx).

## Importante: el conjunto de datos no está en este repositorio

Los scripts necesitan el conjunto de datos multimodal de equipos ágiles de la Universidad de Valparaíso, creado por Diego Miranda. Ese conjunto no se puede publicar aquí, por lo que no se incluye. Debe solicitarse a su autor y guardarse en una carpeta llamada `Root` con esta forma:

```
Root/
    mimicry_indicators.xlsx
    Grupo01/A   Grupo01/B
    Grupo02/A   Grupo02/B   ...
```

Tampoco se incluyen las salidas pesadas (`.npz`, `.parquet`). Sin el conjunto de datos, los scripts no se pueden ejecutar; las tablas de la carpeta `resultados/` permiten revisar lo que se obtuvo.

## Contenido

| Carpeta o archivo | Qué contiene |
|---|---|
| `tp_util.py` | Rutas, argumentos y funciones comunes |
| `tp_cara_01` a `tp_cara_05` (incluye `03b`) | Serie del canal facial (etiqueta `face_mimicry`) |
| `tp_cuerpo_01` a `tp_cuerpo_05` (incluye `03b`) | Serie del canal corporal (etiqueta `body_mimicry`) |
| `tp_sensibilidad_resumen.py` | Reúne los resultados del análisis de sensibilidad |
| `*.bat` | Atajos para Windows con los comandos de cada paso |
| `DECISIONES_cara.md`, `DECISIONES_cuerpo.md` | Registro de decisiones con alternativa y justificación |
| `resultados/` | Salidas livianas de la última prueba (tablas y registros de texto) |
| `etapa_previa/` | Scripts de la etapa anterior (`16g_*`, `util16g.py`), solo como registro |

## 1. Instalar los requisitos

Se necesita Python 3 (probado con 3.13) y estas bibliotecas: numpy, pandas, pyarrow, openpyxl, scipy, scikit-learn, matplotlib, aeon (MiniRocket) y torch (Transformer).

Instalar:

```
pip install numpy pandas pyarrow openpyxl scipy scikit-learn matplotlib aeon torch
```

Actualizar a las últimas versiones:

```
pip install --upgrade numpy pandas pyarrow openpyxl scipy scikit-learn matplotlib aeon torch
```

Para usar la GPU, `torch` debe instalarse con CUDA siguiendo las instrucciones de https://pytorch.org (el comando depende de la versión de CUDA). Sin GPU el Transformer funciona en CPU, pero muy lento. Para comprobar que se detecta la GPU:

```
python -c "import torch; print(torch.cuda.is_available())"
```

Para ver las versiones instaladas: `pip list`.

## 2. Preparar la carpeta de trabajo

Los scripts y la carpeta `Root` van lado a lado:

```
carpeta_de_trabajo/
    Root/                       (el conjunto de datos)
    tp_util.py
    tp_cara_01_entender_datos.py  ...  tp_cuerpo_05_evaluar.py
    tp_sensibilidad_resumen.py
    *.bat
```

Todos los comandos se ejecutan desde esa carpeta:

```
cd ruta\a\carpeta_de_trabajo
```

Si `Root` está en otro lugar, se agrega a cada comando `--root RUTA --anot RUTA\mimicry_indicators.xlsx`. Los resultados quedan en `salidas_tp\` (cara) y `salidas_tp_cuerpo\` (cuerpo).

Cada script retoma donde quedó si se interrumpe: se vuelve a ejecutar el mismo comando. `--rehacer` fuerza todo.

## 3. Preparar los datos (una terminal, en orden)

Canal corporal (se hace primero, porque la etiqueta facial también se predice con series del cuerpo):

```
python tp_cuerpo_01_entender_datos.py
python tp_cuerpo_02_preparar_series.py
python tp_cuerpo_03_construir_casos.py
python tp_cuerpo_03b_seleccionar_puntos.py
```

Canal facial:

```
python tp_cara_01_entender_datos.py
python tp_cara_02_preparar_series.py
python tp_cara_03_construir_casos.py
python tp_cara_03b_seleccionar_puntos.py
```

Atajos: `tp_cuerpo_preparacion.bat` y `tp_cara_preparacion.bat`. El `03b` elige los puntos sin usar etiquetas y deja `tp_*_03b_puntos.csv` con la decisión y el motivo de cada punto.

## 4. Prueba rápida (recomendada)

Con pocos grupos, filtros y épocas. Termina en unos minutos y confirma que los tres modelos funcionan. Sus resultados quedan aparte y se pueden borrar.

```
python tp_cara_04_modelar.py --rep todos --modelos energia sincronia rf minirocket transformer --rapido
```

## 5. Entrenar y comparar los modelos

Cada comando entrena con un grupo fuera (16 pruebas) y guarda los resultados por modelo y representación. Las representaciones son `seleccion` (puntos elegidos sin etiquetas), `todos` y `descriptores`. Los modelos son `energia` y `sincronia` (bases sin aprendizaje), `rf` (Random Forest), `minirocket` y `transformer`.

Modelos de CPU (Random Forest, MiniRocket y bases), una terminal:

```
python tp_cuerpo_04_modelar.py --rep seleccion --modelos energia sincronia rf minirocket
python tp_cara_04_modelar.py --rep seleccion --modelos energia sincronia rf minirocket
```

Transformer (GPU), otra terminal:

```
python tp_cuerpo_04_modelar.py --rep seleccion --modelos transformer
python tp_cara_04_modelar.py --rep seleccion --modelos transformer
```

Para repetir con todas las representaciones, se usan los atajos: `tp_cuerpo_noche_A.bat` y `tp_cara_noche_A.bat` (CPU), y `tp_cuerpo_noche_B.bat` y `tp_cara_noche_B.bat` (GPU). Se recomienda correr a lo más tres procesos a la vez por la memoria.

Etiqueta facial con información del cuerpo (representaciones `cuerpo` y `cara_cuerpo`): `tp_extra_A_gpu.bat` (GPU), `tp_extra_B_cpu.bat` y `tp_extra_C_cpu.bat` (CPU; incluyen la sensibilidad).

Tiempos aproximados (Intel Core i5-12450HX, 16 GB, RTX 3050): el Transformer tarda unos 40 s por grupo de prueba (cerca de 30 minutos por configuración con tres semillas); Random Forest, de 23 s por grupo con el cuerpo a 150 s con cara y cuerpo.

## 6. Evaluar y comparar Random Forest, MiniRocket y Transformer

```
python tp_cuerpo_05_evaluar.py --rep_ref seleccion
python tp_cara_05_evaluar.py --rep_ref seleccion
python tp_sensibilidad_resumen.py --canal cuerpo
python tp_sensibilidad_resumen.py --canal cara
```

La evaluación genera, en cada carpeta de salida:

| Archivo | Para qué sirve |
|---|---|
| `tp_*_05_evaluar.txt` | Resumen completo: si cada modelo supera el azar, comparaciones pareadas y skill por grupo |
| `tp_*_05_resumen_modelos.csv` | Una fila por modelo y representación (PR-AUC, skill, intervalo, valor p) |
| `tp_*_05_comparaciones.csv` | Diferencia de skill entre dos configuraciones, con su intervalo |
| `tp_*_05_por_grupo.csv` | Skill de cada modelo en cada grupo de prueba |
| `tp_*_05_skill_por_grupo.png` | Figura del skill por grupo |
| `tp_*_sensibilidad.txt` y `.csv` | Análisis de sensibilidad |

Para comparar los modelos entre sí, en `tp_*_05_evaluar.txt` se mira:

1. La sección 1 (una fila por modelo): columna `skill_prom` y su intervalo (`skill_ic_inf`, `skill_ic_sup`). Un modelo supera el azar si `supera_azar` es `True`.
2. La sección 2 (comparaciones pareadas): filas como "transformer contra la referencia rf" o "transformer contra minirocket". Una diferencia está respaldada si el intervalo (`ic_inf`, `ic_sup`) no incluye 0 y `p_holm` es menor que 0,05.
3. La sección 3: el skill de cada modelo por grupo, para ver si el resultado depende de pocos grupos.

`skill` vale 0 para el azar y 1 para un puntaje perfecto. `x_azar_prom` (PR-AUC sobre prevalencia) es solo referencial.

## Si algo falla

- Falta de memoria: agregar `--lote_ventanas 500` y, para MiniRocket, `--kernels 5000` (sin mezclar con resultados de 10.000 filtros en la misma carpeta).
- El Transformer informa "CPU (lento)": `torch` no detectó la GPU; revisar la instalación con CUDA.
- Una sesión falla en el script 02: el resumen la lista; se corrige y se vuelve a ejecutar.
