# Plan de Validación — ShopSmart Recommender

**Versión:** 1.0 · **Última actualización:** Etapa 2 del proyecto
**Alcance:** Modelo A (Content-Based) y Modelo B (Popularity), tal como se
entrenaron y evaluaron en `notebooks/03_modeling.ipynb` y
`notebooks/04_evaluation_and_validation.ipynb`.

## 1. Objetivo

Definir **qué se mide, con qué frecuencia, y qué acción dispara cada
resultado**, para que la calidad del sistema de recomendación se pueda
monitorear de forma objetiva a lo largo del tiempo — tanto en la fase
actual (evaluación offline, sin datos de usuario reales) como en una
eventual fase futura con tráfico real (evaluación online).

## 2. Métricas a monitorear

### 2.1 Fase actual — Evaluación offline (sin usuarios reales)

| Métrica | Umbral de alerta | Acción si se cruza el umbral |
|---|---|---|
| Category Precision@10 (Modelo A) | Cae por debajo de **0.70** (vs. 0.824 baseline) | Revisar cambios recientes en el pipeline de features; posible regresión en el `ColumnTransformer` o en los pesos del `TextCombiner` |
| Catalog Coverage@10 (Modelo A) | Cae por debajo de **0.90** (vs. 0.993 baseline) | El modelo está concentrando recomendaciones en un subconjunto chico del catálogo; revisar si `TruncatedSVD` está colapsando dimensiones |
| Avg. Quality@10 (Modelo B) | Cae por debajo de **4.5** (vs. 4.726 baseline) | Revisar la distribución de `bayesian_rating` tras un nuevo scraping; posible entrada de productos de baja calidad sin suficiente evidencia |
| % de valores faltantes tras el notebook 01 | Supera **5%** en cualquier columna crítica (`price_value`, `rating_stars_num`, `asin`) | Bloquear el pipeline; investigar el origen del nuevo scraping antes de reentrenar |
| Cobertura de `asin` entre `products.csv` y `reviews.csv` | Menos del **95%** de productos con al menos 1 reseña | Evaluar si el componente colaborativo-implícito sigue siendo representativo |

Estos umbrales se calculan como *guardrails* relativos al valor obtenido
en el notebook 04 (baseline documentado), no como valores absolutos
arbitrarios — cualquier degradación relevante debe compararse contra esa
línea de base versionada en MLflow (ver sección 5).

### 2.2 Fase futura — Evaluación online (cuando existan datos de usuario reales)

Actualmente el dataset no permite medir engagement real. Si en una
futura iteración se instrumenta la aplicación (Etapa 2, demo Streamlit,
o un e-commerce real), se recomienda capturar como mínimo:

- **CTR (Click-Through Rate)** sobre las recomendaciones mostradas.
- **Add-to-cart rate** desde una recomendación.
- **Tasa de conversión** atribuida a recomendaciones.
- **Tiempo hasta la primera interacción** (proxy de relevancia percibida).

Con estas señales, el plan de validación pasa de métricas proxy
(category precision, coverage) a métricas de negocio directas, y habilita
un **test A/B real** entre el Modelo A y el Modelo B (o variantes de
cada uno), siguiendo el protocolo de la sección 4.

## 3. Frecuencia de evaluación

| Evento | Evaluación requerida |
|---|---|
| Cada `git push` a `main` que modifique `src/`, `notebooks/`, o `requirements.txt` | Suite de tests unitarios (`pytest`) vía GitHub Actions — ver `.github/workflows/ci.yml` |
| Cada nuevo entrenamiento de modelo (`src/pipeline/train_pipeline.py`) | Las 4 métricas de la sección 2.1, registradas automáticamente en MLflow |
| Mensual (o ante cada nuevo scraping del catálogo) | Re-ejecución completa de `01` → `02` → `03` → `04`, comparación contra el run anterior en MLflow |
| Antes de promover un modelo a producción | Revisión manual del `model_evaluation_results.csv` + confirmación de que ningún umbral de la sección 2.1 fue cruzado |

## 4. Protocolo de test A/B (para cuando existan usuarios reales)

1. **Métrica primaria:** tasa de conversión atribuida a recomendaciones
   (o CTR, si el ciclo de conversión es muy largo para medir en el
   horizonte del experimento).
2. **Métricas guardrail** (no deben empeorar aunque la primaria mejore):
   tiempo de carga de la recomendación, tasa de rebote, diversidad de
   categorías vistas por sesión.
3. **Unidad de aleatorización:** sesión de usuario (no producto), para
   evitar contaminación entre variantes dentro de una misma visita.
4. **Tamaño de muestra:** calcular con un análisis de poder estándar
   (α = 0.05, potencia = 0.80) usando la tasa de conversión histórica
   como *baseline* antes de lanzar el experimento — no se define aquí un
   número fijo porque depende del tráfico real, inexistente en esta
   etapa académica.
5. **Duración mínima:** al menos un ciclo semanal completo, para
   controlar variación por día de la semana.

## 5. Versionado y trazabilidad (MLflow)

Cada ejecución de `src/pipeline/train_pipeline.py` registra en MLflow
(ver `mlruns/` local o el servidor configurado):

- **Parámetros:** pesos del `PopularityRecommender`, `min_freq` del
  bucketing de marcas, `n_components` de `TruncatedSVD`, semilla aleatoria.
- **Métricas:** las 4 métricas de la sección 2.1, para ambos modelos.
- **Artefactos:** `feature_engineering_pipeline.joblib`,
  `content_based_recommender.joblib`, `popularity_recommender.joblib`,
  `model_evaluation_results.csv`.

Esto permite comparar cualquier run contra el baseline documentado en el
notebook 04 sin necesidad de volver a ejecutar nada manualmente.

## 6. Criterio de rollback

Si un nuevo modelo entrenado cruza **cualquiera** de los umbrales de la
sección 2.1 frente al run de MLflow marcado como `production` (tag), no
se promueve — se mantiene el modelo vigente y se investiga la causa
(cambio en los datos de entrada, bug en el pipeline, cambio no
intencional de hiperparámetros) antes de reintentar.

## 7. Calidad de datos como precondición de validación

El plan de validación de modelos **depende** de que la calidad de datos
del notebook `01_data_quality_eda.ipynb` se mantenga estable. Por eso la
sección 2.1 incluye umbrales de valores faltantes y cobertura
producto-reseña como parte del mismo plan: un modelo puede degradarse no
por un error propio, sino por un cambio silencioso en la calidad de los
datos de entrada. Esta es la razón por la que ambos notebooks (calidad
de datos y evaluación de modelos) comparten el mismo esquema de umbrales
y alertas.
