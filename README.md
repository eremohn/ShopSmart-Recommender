# ShopSmart Recommender

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://master-7jmeeookyenuvjmnmnvnapp.streamlit.app/)

Sistema de recomendación de productos de moda y accesorios de Amazon,
que combina similitud de contenido con una señal colaborativa-implícita
derivada de reseñas (rating, sentimiento, volumen). Proyecto de
referencia académico para la academia **Henry**.

## 🚀 Demo en vivo

**👉 [Abrir la demo en Streamlit](https://master-7jmeeookyenuvjmnmnvnapp.streamlit.app/)**

Sin instalar nada: elegí un producto del catálogo, compará sus
recomendaciones personalizadas (Content-Based) contra el ranking de
popularidad, y explorá el dashboard de evaluación de modelos.

## 📖 ¿Qué es ShopSmart Recommender?

Un sistema que, para cada producto de un catálogo de 728 artículos,
puede responder dos preguntas distintas:

1. **"Si te gusta este producto, ¿qué más te podría interesar?"** —
   recomendación personalizada por similitud de contenido.
2. **"¿Cuáles son los mejores productos del catálogo?"** — ranking no
   personalizado, útil cuando no hay ningún producto de referencia.

## ⚠️ El problema que condicionó el diseño

`reviews.csv` **no tiene un identificador de usuario/comprador** (solo
`reviewID` y `productASIN`), por lo que un filtrado colaborativo
clásico basado en usuarios no es posible con estos datos. El proyecto
resuelve esto con un enfoque **híbrido basado en producto**: contenido
(marca, categoría, texto) + una señal colaborativa-implícita agregada
por producto (rating bayesiano, sentimiento, volumen de reseñas). El
razonamiento completo de esta decisión está en la
[sección 2 del informe técnico](reports/README.md#2-contexto-y-problema-de-negocio).

## 🧪 Resultado principal

Comparación offline de los dos modelos entrenados (728 productos, K=10):

| Métrica | Content-Based | Popularity (global) |
|---|---:|---:|
| Category Precision@10 | **0.824** | 0.509 |
| Catalog Coverage@10 | **0.993** | 0.014 |
| Avg. Quality@10 | 4.521 | **4.726** |

La diferencia en Category Precision@10 se validó con un test de
Wilcoxon pareado (`p ≈ 1.0 × 10⁻⁸²`, n=728) — es estadísticamente
significativa, no producto del azar. Ningún modelo domina en todas las
métricas: cada uno cumple un rol distinto (motor principal vs.
*fallback* de cold start). Detalle completo, metodología e
interpretación en el informe técnico.

## 📚 Informe técnico

Este README es una guía rápida. Para el análisis completo —
metodología, calidad de datos, EDA con hallazgos e interpretación,
feature engineering, diseño y comparación de los dos modelos,
evaluación estadística, plan de validación, MLflow, y limitaciones—:

**👉 [Consultar el Informe Técnico completo](reports/README.md)**

Documentación complementaria:
- [`docs/VALIDATION_PLAN.md`](docs/VALIDATION_PLAN.md) — plan de
  validación con umbrales de alerta y protocolo de test A/B futuro.
- [`docs/DOCUMENTATION_AUDIT.md`](docs/DOCUMENTATION_AUDIT.md) —
  auditoría de consistencia de la documentación del repositorio.

## 🛠️ Tecnologías utilizadas

`pandas` / `numpy` / `scipy` (análisis y estadística) ·
`scikit-learn` (pipeline de features, `TruncatedSVD`, similitud coseno) ·
`matplotlib` / `seaborn` / `wordcloud` / `missingno` (visualización) ·
`ftfy` (corrección de encoding) · `MLflow` (tracking de experimentos) ·
`Streamlit` (demo funcional) · `pytest` + `GitHub Actions` (testing y CI).

## 🗂️ Dataset

[Amazon E-commerce Product and Review Dataset](https://www.kaggle.com/datasets/lazylad99/amazon-e-commerce-product-and-review-dataset)
(Kaggle): 728 productos (`products.csv`) y 6.327 reseñas (`reviews.csv`)
de ropa y accesorios. Diccionario de datos original en
[`doc/diccionario.txt`](doc/diccionario.txt).

## 📁 Estructura del repositorio

```
ShopSmart-Recommender/
│
├── app.py                      # Demo funcional en Streamlit (raíz, requerida por Streamlit Cloud)
├── requirements.txt
├── mlflow.db                    # Historial de runs de MLflow (ver informe técnico, sección 14)
├── .gitignore
│
├── .github/workflows/ci.yml     # CI: pytest + validación de sintaxis de app.py, en cada push/PR
├── .streamlit/config.toml       # Tema visual de la demo
│
├── config/
│   └── config.yaml               # Parámetros de referencia (no conectado al código, ver auditoría)
│
├── data/
│   ├── raw/                      # products.csv, reviews.csv originales
│   ├── interim/                   # (sin uso actual)
│   ├── processed/                 # Datos limpios, features y resultados de evaluación
│   └── external/                   # (sin uso actual)
│
├── doc/
│   └── diccionario.txt            # Diccionario de datos original del dataset (Kaggle)
│
├── docs/
│   ├── VALIDATION_PLAN.md          # Plan de validación documentado
│   └── DOCUMENTATION_AUDIT.md      # Auditoría de consistencia de la documentación
│
├── notebooks/                     # Notebooks numerados y secuenciales (.ipynb + .py jupytext)
│   ├── 01_data_quality_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation_and_validation.ipynb
│
├── src/                           # Código reutilizable (no exploratorio)
│   ├── data/                       # Carga y limpieza (load_data.py, cleaning.py)
│   ├── features/                    # Transformers y agregaciones reutilizables
│   ├── models/                       # Recomendadores y métricas de evaluación
│   ├── pipeline/                      # Pipeline reproducible end-to-end (MLflow)
│   └── visualization/                  # Paquete reservado, sin contenido aún (ver auditoría)
│
├── models/                         # Modelos entrenados serializados (.joblib)
├── reports/
│   ├── README.md                     # Informe técnico integral (documento maestro)
│   ├── figures/                      # Gráficos exportados desde los notebooks
│   └── powerbi/                       # Reservado para el informe de Power BI (a cargo del equipo)
│
└── tests/                          # 28 tests: cleaning, features, models, pipeline, app (pytest)
```

> La estructura completa fue auditada contra el contenido real del
> repositorio — ver [`docs/DOCUMENTATION_AUDIT.md`](docs/DOCUMENTATION_AUDIT.md)
> para el detalle de qué se verificó y qué inconsistencias se
> encontraron y corrigieron.

## ⚙️ Instalación

```bash
git clone <url-del-repo>
cd ShopSmart-Recommender
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Reproducir el proyecto

### 1. Correr la demo directamente (recomendado)

El repositorio incluye los modelos ya entrenados (`models/*.joblib`) y
los datos procesados (`data/processed/`), por lo que la demo funciona
sin reentrenar nada:

```bash
streamlit run app.py
```

### 2. Reentrenar el pipeline completo

```bash
python -m src.pipeline.train_pipeline
# Con otros hiperparámetros, por ejemplo:
python -m src.pipeline.train_pipeline --w-rating 0.6 --n-components 150
```

Regenera `models/*.joblib` y `data/processed/*`, y registra la corrida
en MLflow.

### 3. Ejecutar los tests

```bash
pytest tests/ -v
```

28 tests: parsing de datos, features, ambos recomendadores y sus
métricas de evaluación, integración del pipeline completo, y smoke
tests de la app de Streamlit.

### 4. Explorar los notebooks paso a paso

```bash
jupyter notebook notebooks/01_data_quality_eda.ipynb
```

Numerados y secuenciales: `01` (calidad de datos + EDA) → `02`
(feature engineering) → `03` (modelado) → `04` (evaluación formal +
plan de validación). El detalle narrativo de cada uno está en el
[informe técnico](reports/README.md).

### 5. Explorar el historial de MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abrir `http://localhost:5000`.

## ☁️ Despliegue de la demo

✅ La demo ya está desplegada en Streamlit Community Cloud (link en la
sección [Demo en vivo](#-demo-en-vivo)), con redespliegue automático en
cada `push` a la rama conectada.

Para desplegar un fork propio: subir el repo a GitHub (asegurándose de
que `app.py`, `requirements.txt`, `models/`, `data/processed/` y
`.streamlit/` queden incluidos — el `.gitignore` de este proyecto ya
está configurado para no excluirlos), entrar a
[share.streamlit.io](https://share.streamlit.io), iniciar sesión con
GitHub, y crear una nueva app apuntando a `app.py` en la rama `main`.

## 👤 Autoría

Proyecto de referencia elaborado por el/la tutor/a de la academia
**Henry** como ejemplo guía del proyecto final.
