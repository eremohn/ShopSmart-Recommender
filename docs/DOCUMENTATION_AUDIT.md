# Auditoría de Documentación — ShopSmart Recommender

**Fecha de auditoría:** realizada sobre el estado real del repositorio
al momento de escribir este documento (código, notebooks ejecutados,
artefactos serializados, `mlflow.db`, tests).

**Metodología:** se inspeccionó directamente el contenido del
repositorio (no la documentación previa ni el historial de conversación)
y se verificó cada cifra citada en `README.md` y
`reports/TECHNICAL_REPORT.md` contra su fuente real: CSVs crudos y
procesados, outputs cacheados de los notebooks ejecutados,
`model_evaluation_results.csv`, y `mlflow.db`. Este documento reporta
únicamente hallazgos verificables.

## 1. Inconsistencias encontradas

### 1.1 `products_master.csv` tiene más columnas de las que documenta el notebook 02

**Hallazgo:** la celda de salida cacheada del notebook
`02_feature_engineering.ipynb` (ejecutada la última vez que el notebook
corrió de forma aislada) reporta que `products_master.csv` se exportó
con **16 columnas** (`df_master[master_export_cols]`, un subconjunto
curado). Sin embargo, el archivo real actualmente en
`data/processed/products_master.csv` tiene **58 columnas**.

**Causa raíz:** `src/pipeline/train_pipeline.py::run_pipeline()`
(construido después del notebook 02, como parte de la Etapa 2) guarda
`df_master` completo con `df_master.to_csv(...)` — todas sus columnas,
no el subconjunto curado de 16 que arma manualmente el notebook. Cada
vez que se corrió el pipeline reproducible después de haber corrido el
notebook 02, el archivo en disco quedó con el esquema del pipeline
(58 columnas), sobrescribiendo el del notebook.

**Impacto:** ninguno funcional — `app.py` y el resto del código
solo leen columnas específicas por nombre (`title`, `brand_name`,
`main_category`, `price_value_winsorized`, `bayesian_rating`, etc.), que
están presentes en ambas versiones del archivo. El impacto es puramente
documental: alguien que lea la celda de salida del notebook 02 y luego
inspeccione el CSV real vería un número de columnas distinto.

**Decisión tomada en esta auditoría:** no se resolvió re-ejecutando el
notebook 02 (habría alterado un artefacto ya validado en las secciones 9
y 12 del informe técnico, con riesgo de introducir nuevas
inconsistencias). En su lugar, se documenta explícitamente esta
diferencia en `reports/TECHNICAL_REPORT.md` (sección 7.2) y aquí. El
archivo real de 58 columnas es un superset del de 16 — no hay pérdida de
información, solo una diferencia de alcance entre "lo que un notebook
exploratorio decide exportar" y "lo que un pipeline de producción
persiste por completitud".

**Recomendación pendiente:** si se prioriza que ambos coincidan
exactamente, alinear `run_pipeline()` para que exporte el mismo
subconjunto curado de columnas que el notebook 02, o —alternativa
preferible— eliminar la exportación manual del notebook 02 y declarar a
`src/pipeline/train_pipeline.py` como la única fuente de verdad para
este artefacto.

### 1.2 Nulos residuales en `product_description` y `customer_review_summary` tras la exportación a CSV

**Hallazgo:** el notebook 01 imputa estos dos campos con cadena vacía
(`""`), y su celda de salida reporta "0 valores nulos restantes" en
`reviews_clean.csv` y un conteo bajo en `products_clean`. Al releer
`data/processed/products_clean.csv` desde disco, sin embargo,
`product_description` tiene 457 filas `NaN` y `customer_review_summary`
tiene 94.

**Causa raíz:** comportamiento estándar de `pandas.read_csv`, que
interpreta una cadena vacía `""` en un CSV como valor faltante (`NaN`)
por defecto al releer el archivo, aunque el DataFrame en memoria, antes
de exportar, tuviera efectivamente `""` y no `NaN`.

**Impacto:** ninguno funcional — en `TextCombiner.transform()`
(`src/features/transformers.py`) el manejo es `str(row[col]) if
pd.notna(row[col]) else ""`, es decir, un `NaN` releído se convierte de
vuelta a `""` exactamente como se pretendía. El comportamiento final es
idéntico con o sin este artefacto de re-lectura.

**Decisión tomada:** documentar el comportamiento (aquí y en la sección
7.2 del informe técnico) en vez de "corregirlo", porque no hay nada que
corregir en el resultado funcional — es una característica conocida de
`pandas.read_csv`, no un error del pipeline.

### 1.3 Docstring de `train_pipeline.py` con referencia obsoleta a `./mlruns`

**Hallazgo:** el módulo `src/pipeline/train_pipeline.py` usaba, en su
docstring de módulo, el ejemplo `mlflow ui  # para explorar los runs
registrados en ./mlruns`, pero el código real configura
`mlflow.set_tracking_uri("sqlite:///mlflow.db")` — no usa `./mlruns`.

**Causa raíz:** el backend de tracking se migró de archivos planos
(`./mlruns`) a SQLite durante el desarrollo (MLflow ≥ 3.x deprecó el
backend de archivos plano), pero el comentario de ejemplo en el
docstring no se había actualizado.

**Estado:** **corregido** como parte de esta auditoría. El docstring
ahora dice `mlflow ui --backend-store-uri sqlite:///mlflow.db`,
coincidiendo con el código real.

### 1.4 Comentario de tamaño de datos desactualizado en `.gitignore`

**Hallazgo:** el comentario explicativo en `.gitignore` sobre por qué
`data/` y `models/` se versionan decía *"Son datasets públicos y
pequeños (~5 MB en total)"*. La medición real es `data/` ≈ 16 MB +
`models/` ≈ 8.1 MB ≈ 24 MB combinados.

**Estado:** **corregido** como parte de esta auditoría. El comentario
ahora dice *"~15 MB los datos + ~8 MB los modelos serializados"*,
verificado contra el tamaño real de ambos directorios.

### 1.5 `config/config.yaml` no está conectado a ningún código

**Hallazgo:** existe `config/config.yaml` con rutas de datos, semilla
aleatoria y parámetros de calidad de datos (`outlier_method`,
`iqr_multiplier`, `low_review_count_threshold`). Se verificó, mediante
búsqueda exhaustiva (`grep` de `"config.yaml"`, `"yaml.safe_load"`,
`"import yaml"`) en `src/`, `notebooks/*.py` y `app.py`, que **ningún
archivo de código lee este archivo**. Los notebooks y `src/pipeline/train_pipeline.py`
hardcodean sus propias rutas y valores por defecto (algunos de los
cuales coinciden con `config.yaml`, como `random_seed: 42`, y otros no
— por ejemplo, `config.yaml` declara `iqr_multiplier: 1.5`, pero el
notebook 01 no expone ese multiplicador como parámetro configurable en
su implementación real del método IQR).

**Impacto:** bajo, pero real — `config/config.yaml` puede dar la falsa
impresión de ser la fuente de configuración del proyecto cuando en la
práctica no lo es.

**Recomendación pendiente (no resuelta en esta auditoría, por alcance):**
o bien conectar `config.yaml` al pipeline reproducible (por ejemplo,
permitiendo que `train_pipeline.py` lea sus valores por defecto desde
ahí en lugar de tenerlos hardcodeados en `argparse`), o bien eliminarlo
y documentar los parámetros únicamente donde se usan (`src/pipeline/train_pipeline.py`,
sección 11.3 del informe técnico). Se optó por **no eliminarlo ni
modificar su rol** en esta auditoría para no alterar comportamiento
existente sin una decisión explícita del equipo del proyecto.

### 1.6 `src/visualization/` es un paquete vacío

**Hallazgo:** `src/visualization/` contiene únicamente un
`__init__.py` vacío (0 bytes). No hay ningún módulo de funciones de
ploteo reutilizables dentro de esta carpeta — toda la lógica de
visualización de los notebooks (`save_fig()`, llamadas a `matplotlib`/
`seaborn`/`wordcloud`) está definida directamente dentro de cada
notebook, no en `src/`.

**Impacto:** ninguno funcional (el paquete no rompe nada al estar
vacío), pero es una carpeta que promete una responsabilidad
("visualización reutilizable") que no cumple actualmente.

**Recomendación pendiente:** o se completa con las funciones de
ploteo que hoy están duplicadas entre notebooks (por ejemplo, `save_fig()`
se redefine de forma idéntica en los 4 notebooks), o se elimina la
carpeta para no sugerir una capacidad inexistente.

## 2. Situaciones documentadas (no son errores, pero requieren aclaración)

### 2.1 Relación entre los archivos `.ipynb` y `.py` de cada notebook

**Verificación realizada:** se regeneró cada `.py` a partir de su
`.ipynb` correspondiente con `jupytext --to py:percent` y se comparó
contra el `.py` real del repositorio. La única diferencia encontrada en
los 4 notebooks fue en los metadatos de cabecera de `jupytext`
(`format_version`, `jupytext_version`) — **el código y las celdas
markdown son idénticos** entre ambas representaciones.

**Cómo se relacionan realmente:** el archivo `.py` (formato `jupytext`
*percent*, sin outputs) es la fuente que se escribió primero; el
`.ipynb` se generó una vez a partir de él (`jupytext --to ipynb`) y
luego se ejecutó (`jupyter nbconvert --execute`) para producir los
outputs y gráficos ya horneados en el `.ipynb`. **No existe un
mecanismo de sincronización automática**: no hay metadata de
`jupytext: formats: ipynb,py:percent` en la cabecera de ninguno de los
dos archivos, por lo que abrir y editar el `.ipynb` directamente en
Jupyter (o Colab) **no actualizaría** el `.py` correspondiente, y
viceversa.

**Riesgo de mantenimiento:** si en el futuro alguien edita el `.ipynb`
sin regenerar el `.py` (o al revés), ambos archivos divergerán en
silencio, sin ninguna advertencia automática. El repositorio no tiene
actualmente ningún test o chequeo de CI que detecte esta divergencia.

**Recomendación pendiente:** agregar la configuración de pareo de
`jupytext` (`jupytext --set-formats ipynb,py:percent notebooks/*.ipynb`)
para que ambas representaciones se sincronicen automáticamente al
guardar desde Jupyter, o agregar un chequeo de CI que regenere el `.py`
desde el `.ipynb` y falle si hay diferencias de contenido (ignorando
metadata de versión).

### 2.2 `doc/` vs. `docs/` — no son redundantes, pero el parecido de nombres genera confusión

**Verificación realizada:** se inspeccionó el contenido completo de
ambas carpetas.

- **`doc/`** (singular): contiene únicamente `diccionario.txt` — el
  diccionario de datos **provisto originalmente junto con el dataset**
  (descripción en inglés de cada columna de `products.csv` y
  `reviews.csv`). Es un documento de referencia externo al proyecto, no
  generado por el equipo.
- **`docs/`** (plural): contiene `VALIDATION_PLAN.md` y (a partir de
  esta auditoría) `DOCUMENTATION_AUDIT.md` — documentación **producida
  durante el desarrollo del proyecto**.

**Conclusión de la auditoría:** no hay redundancia de contenido — cada
carpeta tiene un propósito distinto y ningún archivo se repite entre
ambas. El problema real es puramente de **nomenclatura**: dos carpetas
con nombres casi idénticos (`doc` / `docs`) en la raíz del repositorio
es una fuente previsible de confusión para quien navegue el repo por
primera vez, incluso si su contenido no se superpone.

**Recomendación pendiente (no ejecutada en esta auditoría):** renombrar
`doc/` a algo más explícito, por ejemplo `data/dictionary/` o
`docs/dataset/diccionario.txt` (fusionando físicamente ambas carpetas
bajo `docs/`, con subcarpetas si crece el contenido). No se ejecutó este
renombre en la presente auditoría porque `doc/diccionario.txt` está
referenciado por ruta relativa desde `README.md` y
`reports/TECHNICAL_REPORT.md`; moverlo requeriría actualizar esas
referencias de forma coordinada, y se prefirió no introducir ese cambio
estructural sin confirmación explícita del equipo del proyecto.

### 2.3 `.devcontainer/` no existe en el repositorio

Se verificó explícitamente (`find .devcontainer -type f`) que esta
carpeta, mencionada como punto de auditoría posible, **no existe** en el
estado actual del repositorio. No se documenta más allá de esta
constancia, para evitar dar la impresión de que falta algo que nunca
formó parte del proyecto.

### 2.4 `reports/powerbi/` está vacía

Se verificó que la carpeta existe pero no contiene archivos. Es
intencional: el informe de Power BI está **a cargo del equipo del
curso**, fuera del alcance de este proyecto de referencia (así lo
indican tanto `README.md` como este informe). No se trata como una
inconsistencia, sino como un placeholder documentado de un entregable
externo pendiente.

## 3. Verificaciones realizadas (evidencia de que las cifras citadas son reales)

Como parte de esta auditoría se ejecutaron, entre otras, las siguientes
verificaciones directas sobre el repositorio (no sobre documentación
previa):

- Lectura directa de `data/raw/products/products.csv` y
  `data/raw/reviews/reviews.csv` con `pandas` para confirmar shape,
  duplicados, e integridad referencial (`asin` ↔ `productASIN`).
- Cálculo directo de porcentajes de valores faltantes por columna sobre
  los CSVs crudos (no reutilizado de ninguna fuente anterior).
- Verificación de que los 4 notebooks tienen 0 celdas con errores de
  ejecución (`output_type == "error"`) inspeccionando el JSON de cada
  `.ipynb`.
- Extracción de las cifras de resultados (Category Precision@10,
  Catalog Coverage@10, Intra-list Diversity@10, Avg. Quality@10, p-value
  de Wilcoxon, intervalos de confianza bootstrap) directamente de los
  outputs cacheados de `notebooks/03_modeling.ipynb` y
  `notebooks/04_evaluation_and_validation.ipynb`, y cruzadas contra
  `data/processed/model_evaluation_results.csv` y las métricas
  registradas en `mlflow.db` — las tres fuentes coinciden exactamente.
- Ejecución real de `pytest tests/ -q`: 28/28 tests pasan sobre el
  estado actual del repositorio.
- Verificación de que todos los enlaces relativos (`../notebooks/...`,
  `../src/...`, `figures/...`, etc.) usados en
  `reports/TECHNICAL_REPORT.md` apuntan a archivos que existen
  realmente en el repositorio.

## 4. Cambios realizados como parte de esta auditoría

1. Creado `reports/TECHNICAL_REPORT.md` — informe técnico integral (25
   secciones), con todas las cifras verificadas contra el repositorio
   real.
2. Reescrito `README.md` — portada del proyecto, ahora enlaza al informe
   técnico y refleja la estructura real y verificada del repositorio.
3. Corregido el docstring de `src/pipeline/train_pipeline.py`
   (referencia obsoleta a `./mlruns`, sección 1.3).
4. Corregido el comentario de tamaño de datos en `.gitignore` (sección
   1.4).
5. Corregidos dos enlaces rotos en `app.py`: un link a `https://github.com/`
   sin URL real de destino en el sidebar, y otro con el mismo problema
   apuntando nominalmente a `docs/VALIDATION_PLAN.md` en la página de
   evaluación. Al no contar con una URL de repositorio de GitHub
   verificable para apuntar, se optó por quitar el hipervínculo roto y
   dejar la referencia como texto plano, en vez de dejar un link que no
   funciona.
6. Creado este documento, `docs/DOCUMENTATION_AUDIT.md`.

## 5. Recomendaciones pendientes (no ejecutadas, requieren decisión del equipo)

Listadas en orden de impacto estimado, de mayor a menor:

1. **Alinear el esquema de `products_master.csv`** entre el notebook 02
   y `train_pipeline.py` (sección 1.1), o declarar explícitamente cuál
   de los dos es la fuente de verdad para este artefacto.
2. **Configurar el pareo automático de `jupytext`** entre `.ipynb` y
   `.py`, o agregar un chequeo de CI que detecte divergencia (sección
   2.1) — es el hallazgo con mayor riesgo de causar inconsistencias
   futuras si el proyecto sigue evolviendo.
3. **Decidir el rol de `config/config.yaml`**: conectarlo al código o
   eliminarlo (sección 1.5).
4. **Completar o eliminar `src/visualization/`** (sección 1.6).
5. **Resolver el parecido de nombres `doc/` / `docs/`** (sección 2.2),
   coordinando la actualización de las referencias relativas si se
   decide fusionar o renombrar.
6. **Evaluar la migración de `mlflow.db` a un backend remoto** si el
   proyecto pasa a tener más de un colaborador entrenando en paralelo
   (ya documentado con el análisis a favor/en contra en
   `reports/TECHNICAL_REPORT.md`, sección 14.6).
