# deteccion-mimetismo-equipos-agiles

Scripts para entrenar y validar un modelo de detección de mimetismo corporal y facial en equipos ágiles, a partir de los keypoints del conjunto de datos multimodal de la Universidad de Valparaíso. Proyecto de título, Ingeniería Civil Informática.

## Qué hace

Dado el conjunto de datos de equipos ágiles (19 grupos de cuatro estudiantes, 16 con mimetismo anotado), el código:

1. Limpia y alinea las series de puntos clave (pose, manos, cara) y las divide en segmentos de 2 segundos.
2. Reconoce la acción de cada persona en cada segmento (etapa 1, *Random Forest*).
3. Puntúa cada par de personas por la probabilidad de que hagan la misma acción a la vez (etapa 2, sin entrenamiento).
4. Valida con la técnica **un grupo fuera** (se entrena con 15 grupos y se prueba con el restante, 16 veces).
5. Agrega datos sintéticos de acciones corporales al entrenamiento, y guarda un modelo final entrenado con los 16 grupos.

**Criterio de éxito:** PR-AUC al menos 2 veces la prevalencia (el azar), con el límite inferior del intervalo de confianza de 95 % sobre la prevalencia.

## Resultado esperado

| Canal | Sin sintéticos | Con 3.000 sintéticos por acción | Criterio |
|---|---|---|---|
| Cuerpo | 1,68 veces el azar | **2,01** (2,00 en una repetición con otra semilla) | cumple, por un margen mínimo |
| Cara | 6,80 veces el azar | **6,26** | cumple |

Con 5.000 sintéticos por acción el cuerpo llega a 2,09 y la cara baja a 5,93. Los datos sintéticos mejoran el cuerpo pero empeoran algo la cara, y el resultado del cuerpo es optimista porque la cantidad se eligió mirando los mismos grupos de prueba. El detalle está en el informe del proyecto.

## Dónde dejar cada cosa

El conjunto de datos **no está en este repositorio**: debe descargarse desde su fuente original (Miranda et al., 2025). Se guarda en una carpeta llamada `Root`, **al lado de los scripts**, dentro de una misma carpeta de trabajo:

```
carpeta_de_trabajo/
│
├── Root/                          <- el conjunto de datos descargado
│   ├── mimicry_indicators.xlsx
│   ├── Grupo01/
│   │   ├── A/
│   │   └── B/
│   ├── Grupo02/
│   │   ├── A/
│   │   └── B/
│   └── ...                        (hasta Grupo21)
│
├── 16g_00_comprobar_datos.py      <- los scripts de este repositorio
├── 16g_01_preparar_series.py
├── ...
├── 16g_12_modelo_final.py
├── util16g.py
│
└── salidas_16g/                   <- se crea sola al ejecutar; aquí quedan los resultados
```

Si el conjunto de datos está en otro lugar, se indica con `--root RUTA` y `--anot RUTA\mimicry_indicators.xlsx` en cada comando.

## Requisitos

Python 3 y estas bibliotecas (probado con Python 3.13, numpy 2.5.5, pandas 3.1.1, pyarrow 26.0.1, scikit-learn 1.9.1, matplotlib 3.11.2, en Windows 11 con 12 procesadores lógicos):

```
pip install pandas pyarrow openpyxl numpy scikit-learn matplotlib joblib
```

## Paso a paso

1. Descargar el repositorio (botón verde *Code* > *Download ZIP*, o `git clone`) y descomprimirlo en la `carpeta_de_trabajo`.
2. Descargar el conjunto de datos y dejarlo como `Root`, al lado de los scripts (ver el esquema de arriba).
3. Abrir la consola (*cmd*) **en la carpeta donde están los scripts**: en el Explorador de archivos, escribir `cmd` en la barra de direcciones y presionar Enter, o usar `cd ruta\a\carpeta_de_trabajo`.
4. Ejecutar los comandos de la siguiente sección, **en este orden** (cada script usa las salidas del anterior).

### Camino directo al modelo final

| Paso | Comando | Qué hace | Dónde ver el resultado |
|---|---|---|---|
| 1 | `python 16g_00_comprobar_datos.py` | Comprueba que estén las 32 sesiones y sus archivos | `salidas_16g/16g_00_comprobar_datos.txt` |
| 2 | `python 16g_01_preparar_series.py` | Limpia y alinea las series de puntos clave | `salidas_16g/16g_01_series_<sesión>.parquet` |
| 3 | `python 16g_02_construir_casos.py` | Arma los casos segmento-par-canal con su etiqueta | `salidas_16g/16g_02_casos.parquet` |
| 4 | `python 16g_05_dos_etapas_logo.py` | Características por persona y caso base sin sintéticos | `salidas_16g/16g_05_resultados.csv`; el caso base (cuerpo 1,68 y cara 6,80 veces el azar) se ve también como fila `-1` en el paso 5 |
| 5 | `python 16g_10_representacion_fina.py --rep base --objetivos 3000 5000 --max-por-ejemplo 40` | Genera los datos sintéticos y valida con un grupo fuera | `salidas_16g/16g_10_base_resultados.csv`: filas con objetivo 3000 y 5000, columna `cumple_global` en `True` para el cuerpo |
| 6 | `python 16g_12_modelo_final.py` | Entrena el modelo final con los 16 grupos y lo guarda | `salidas_16g/16g_12_modelo_final.joblib` y `16g_12_ficha_modelo.txt` |
| 7 | `python 16g_12_modelo_final.py --modo puntuar --grupos 01 15` | Puntúa los pares de los grupos indicados con el modelo guardado | `16g_12_candidatos_puntuar.csv` (segmentos-par con mayor puntaje) |

Usar exactamente la orden del paso 5 y la semilla por defecto (0): la cantidad máxima de la reserva de sintéticos cambia las variantes generadas, y con otra orden el resultado varía levemente.

### Cómo comprobar que se cumple el umbral

- Abrir `salidas_16g/16g_10_base_resultados.csv`. Las filas con `objetivo` 3000 y 5000 deben mostrar `cumple_global = True` en el canal `cuerpo` (columna `x_azar` cerca de 2,01 y 2,09) y en el canal `cara` (cerca de 6,26 y 5,93). La fila `-1` es el caso sin sintéticos.
- `salidas_16g/16g_10_base_por_grupo.csv` muestra el resultado de cada grupo por separado.
- `salidas_16g/16g_12_ficha_modelo.txt` resume la configuración del modelo final y copia ese desempeño validado.

### Comprobar que todo salió bien

**1. Ver que se cumple el criterio.** Muestra, por canal y cantidad de sintéticos, las veces sobre el azar y si se cumple:

```
python -c "import pandas as pd; d=pd.read_csv('salidas_16g/16g_10_base_resultados.csv'); print(d[['objetivo','canal','x_azar','cumple_global']])"
```

Debe mostrar `cumple_global = True` en cuerpo y cara con objetivo 3000 y 5000 (cuerpo cerca de 2,01 y 2,09; cara cerca de 6,26 y 5,93). La fila `-1` es el caso sin datos sintéticos.

**2. Abrir el modelo guardado.** Muestra su configuración, las acciones que reconoce y el número de características (178):

```
python -c "import joblib; M=joblib.load('salidas_16g/16g_12_modelo_final.joblib'); print(M['configuracion']); print(list(M['modelo'].classes_)); print(len(M['caracteristicas']), 'caracteristicas')"
```

El criterio se verifica con la primera orden y no con el modelo guardado: el modelo final ya vio los 16 grupos y no tiene un conjunto de prueba propio.

### Sobre el modelo final

El modelo guardado se entrena con los 16 grupos, así que **no tiene un conjunto de prueba propio**. El desempeño que se le atribuye es el de la validación con un grupo fuera (paso 5). El puntaje que el script calcula sobre esos mismos 16 grupos sale muy alto y no es una evaluación. Un mismo modelo sirve a los dos canales: el puntaje del par se contrasta con la anotación corporal o facial.

## Reproducir el resto de los resultados del informe

Siempre después de los pasos 1 a 4:

| Comando | Resultado que reproduce |
|---|---|
| `python 16g_03_medidas_sincronia.py` y luego `python 16g_04_modelos_clasicos_logo.py` | Medidas clásicas de sincronía y modelos clásicos (no cumplen el criterio) |
| `python 16g_06_resumen_y_figuras.py` | Resumen global y por grupo, y las figuras |
| `python 16g_07_diagnostico_actividad.py` y luego `python 16g_08_diagnostico_etapa1.py` | Diagnósticos de la regla de la misma acción y del reconocimiento de acciones |
| `python 16g_09_datos_sinteticos_cuerpo.py` | Sintéticos con 500, 1.000 y 2.000 ejemplos por acción |
| `python 16g_10_representacion_fina.py --rep fina --objetivos 0 1000 2000` | Características finas de las manos, sin y con sintéticos |
| `python 16g_10_representacion_fina.py --rep base --objetivos 3000 --semilla 1` | Repetición de la configuración final con otra semilla (2,00 veces el azar) |
| `python 16g_11_comparacion_pareada.py` | Comparación pareada de cada variante con el caso base |

Para una prueba rápida con dos grupos, agregar `--grupos 01 15` a los scripts 00 a 03.

## Límites

- Los resultados corresponden a 16 grupos de estudiantes de una universidad, en laboratorio; no se extienden a otros equipos ni tareas.
- El mimetismo "facial" detectado es en buena parte reconocimiento de acciones (mano en la boca, apoyarse hacia atrás), porque el canal en que se anota el mimetismo depende del tipo de acción.
- Para puntuar sesiones nuevas deben calcularse antes sus características (pasos 1 a 4).

## Cita del conjunto de datos

Miranda et al. (2025). Conjunto de datos multimodal de equipos ágiles de la Universidad de Valparaíso (*Data in Brief*).
