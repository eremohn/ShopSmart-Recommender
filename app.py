"""ShopSmart Recommender — Demo funcional en Streamlit.

Expone los dos modelos entrenados en `notebooks/03_modeling.ipynb`
(`ContentBasedRecommender` y `PopularityRecommender`) a través de una
interfaz interactiva, además de un dashboard con los resultados de la
evaluación formal del notebook `04_evaluation_and_validation.ipynb`.

Ejecución local
----------------
```bash
pip install -r requirements.txt
streamlit run app.py
```

Esta app está pensada para desplegarse en Streamlit Community Cloud
apuntando directamente a este archivo — ver instrucciones en el README.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

st.set_page_config(
    page_title="ShopSmart Recommender",
    page_icon="🛍️",
    layout="wide",
)


# --------------------------------------------------------------------------
# Carga de artefactos (cacheada: se ejecuta una sola vez por sesión/deploy)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelos entrenados...")
def load_models():
    content_model = joblib.load(MODELS_DIR / "content_based_recommender.joblib")
    popularity_model = joblib.load(MODELS_DIR / "popularity_recommender.joblib")
    return content_model, popularity_model


@st.cache_data(show_spinner="Cargando catálogo de productos...")
def load_products_master() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "products_master.csv")


@st.cache_data(show_spinner=False)
def load_evaluation_results() -> pd.DataFrame:
    path = PROCESSED_DIR / "model_evaluation_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


def _missing_artifacts_message() -> None:
    st.error(
        "No se encontraron los artefactos entrenados (`models/*.joblib`, "
        "`data/processed/products_master.csv`). Ejecutá primero el pipeline "
        "reproducible antes de correr la demo:\n\n"
        "```bash\npython -m src.pipeline.train_pipeline\n```"
    )
    st.stop()


try:
    content_model, popularity_model = load_models()
    products_master = load_products_master()
except FileNotFoundError:
    _missing_artifacts_message()


# --------------------------------------------------------------------------
# Sidebar — navegación
# --------------------------------------------------------------------------
st.sidebar.title("🛍️ ShopSmart Recommender")
st.sidebar.caption("Proyecto de referencia — Academia Henry")
page = st.sidebar.radio(
    "Navegación",
    ["🔍 Explorar y Recomendar", "📊 Evaluación de Modelos", "ℹ️ Acerca del Proyecto"],
)
st.sidebar.divider()
st.sidebar.markdown(
    "**Dataset:** [Amazon E-commerce Product & Review](https://www.kaggle.com/datasets/lazylad99/amazon-e-commerce-product-and-review-dataset)"
)


# --------------------------------------------------------------------------
# Página 1 — Explorar y Recomendar
# --------------------------------------------------------------------------
def render_product_card(row: pd.Series) -> None:
    with st.container(border=True):
        st.markdown(f"**{row['title'][:90]}**")
        cols = st.columns(4)
        cols[0].metric("Marca", row["brand_name"])
        cols[1].metric("Categoría", row["main_category"])
        cols[2].metric("Precio", f"${row['price_value_winsorized']:.2f}")
        cols[3].metric("Rating", f"{row['rating_stars_num']:.1f} ⭐")


def render_recommendations_table(asins: list[str], score_col: str, score_label: str) -> None:
    df = products_master.set_index("asin").loc[asins].reset_index()
    display_cols = {
        "title": "Producto",
        "brand_name": "Marca",
        "main_category": "Categoría",
        "price_value_winsorized": "Precio",
        "bayesian_rating": "Rating (bayesiano)",
    }
    df_display = df[list(display_cols.keys())].rename(columns=display_cols)
    df_display["Precio"] = df_display["Precio"].map(lambda x: f"${x:.2f}")
    df_display["Rating (bayesiano)"] = df_display["Rating (bayesiano)"].map(lambda x: f"{x:.2f}")
    st.dataframe(df_display, width="stretch", hide_index=True)


if page == "🔍 Explorar y Recomendar":
    st.title("🔍 Explorar productos y generar recomendaciones")
    st.write(
        "Elegí un producto del catálogo para ver recomendaciones "
        "**personalizadas** (Modelo A — Content-Based) y compararlas con "
        "el ranking **no personalizado** de mejores productos "
        "(Modelo B — Popularity)."
    )

    categories = ["Todas"] + sorted(products_master["main_category"].unique().tolist())
    selected_category = st.selectbox("Filtrar por categoría", categories)

    filtered = products_master
    if selected_category != "Todas":
        filtered = filtered[filtered["main_category"] == selected_category]

    product_titles = filtered.set_index("asin")["title"].to_dict()
    selected_asin = st.selectbox(
        "Producto de referencia",
        options=list(product_titles.keys()),
        format_func=lambda asin: product_titles[asin][:80],
    )

    k = st.slider("Cantidad de recomendaciones (K)", min_value=3, max_value=15, value=5)

    st.subheader("Producto consultado")
    render_product_card(products_master.set_index("asin").loc[selected_asin])

    tab_content, tab_popularity, tab_popularity_cat = st.tabs(
        [
            "🎯 Content-Based (personalizado)",
            "🔥 Popularity (global)",
            "🔥 Popularity (misma categoría)",
        ]
    )

    with tab_content:
        st.caption(
            "Recomendaciones basadas en similitud de contenido (marca, "
            "categoría, texto del producto y de las reseñas)."
        )
        recs = content_model.recommend(selected_asin, k=k)
        render_recommendations_table(recs["asin"].tolist(), "similarity_score", "Similitud")

    with tab_popularity:
        st.caption(
            "Ranking global de \"lo mejor del catálogo\" — **no cambia** "
            "según el producto consultado (es el mismo para cualquier usuario)."
        )
        recs = popularity_model.recommend(k=k)
        render_recommendations_table(recs["asin"].tolist(), "popularity_score", "Popularidad")

    with tab_popularity_cat:
        query_category = products_master.set_index("asin").loc[selected_asin, "main_category"]
        st.caption(f"Ranking de mejores productos dentro de la categoría **{query_category}**.")
        recs = popularity_model.recommend(k=k + 1, main_category=query_category)
        recs = recs[recs["asin"] != selected_asin].head(k)
        render_recommendations_table(recs["asin"].tolist(), "popularity_score", "Popularidad")


# --------------------------------------------------------------------------
# Página 2 — Evaluación de Modelos
# --------------------------------------------------------------------------
elif page == "📊 Evaluación de Modelos":
    st.title("📊 Evaluación formal de los modelos")
    st.write(
        "Resultados de la comparación offline documentada en "
        "`notebooks/03_modeling.ipynb` y `notebooks/04_evaluation_and_validation.ipynb` "
        "(728 productos, K=10). La significancia estadística de la "
        "diferencia entre modelos se validó con un test de Wilcoxon "
        "pareado (p ≈ 1.0 × 10⁻⁸²)."
    )

    df_eval = load_evaluation_results()
    if df_eval.empty:
        st.warning("No se encontró `model_evaluation_results.csv`. Corré el pipeline primero.")
    else:
        st.dataframe(df_eval.round(3), width="stretch")

        metric_cols = st.columns(len(df_eval.columns))
        for col, metric in zip(metric_cols, df_eval.columns):
            with col:
                st.markdown(f"**{metric}**")
                st.bar_chart(df_eval[metric])

    st.divider()
    st.subheader("Umbrales de alerta (Plan de Validación)")
    st.markdown(
        """
        | Métrica | Baseline | Umbral de alerta |
        |---|---|---|
        | Category Precision@10 (Content-Based) | 0.824 | < 0.70 |
        | Catalog Coverage@10 (Content-Based) | 0.993 | < 0.90 |
        | Avg. Quality@10 (Popularity) | 4.726 | < 4.50 |

        Detalle completo, protocolo de test A/B y criterio de rollback en
        `docs/VALIDATION_PLAN.md` (repositorio del proyecto).
        """
    )

    figure_path = FIGURES_DIR / "22_bootstrap_confidence_intervals.png"
    if figure_path.exists():
        st.subheader("Intervalos de confianza (bootstrap, 95%)")
        st.image(str(figure_path), width="stretch")


# --------------------------------------------------------------------------
# Página 3 — Acerca del Proyecto
# --------------------------------------------------------------------------
else:
    st.title("ℹ️ Acerca de ShopSmart Recommender")
    st.markdown(
        """
        Sistema de recomendación híbrido de productos de moda y accesorios
        de Amazon, desarrollado como **proyecto de referencia académico**
        para la academia **Henry**.

        ### Decisión de diseño clave
        `reviews.csv` no incluye un identificador de usuario/comprador, lo
        que impide un filtrado colaborativo clásico basado en usuarios. El
        proyecto adopta en su lugar un enfoque **híbrido basado en
        producto**:

        - **Modelo A — Content-Based**: recomendación personalizada por
          similitud de contenido (marca, categoría, texto).
        - **Modelo B — Popularity**: ranking no-personalizado basado en
          rating bayesiano, sentimiento y volumen de reseñas — aproxima el
          componente "colaborativo-implícito".

        ### Estructura del proyecto

        ```
        notebooks/
          01_data_quality_eda.ipynb          Calidad de datos + EDA
          02_feature_engineering.ipynb       Pipeline de features (sklearn)
          03_modeling.ipynb                  Entrenamiento y comparación
          04_evaluation_and_validation.ipynb Evaluación formal + estadística
        src/
          data/       Carga y limpieza de datos
          features/   Transformers y agregaciones reutilizables
          models/     Recomendadores y métricas de evaluación
          pipeline/   Pipeline reproducible end-to-end (MLflow)
        docs/
          VALIDATION_PLAN.md                 Plan de validación documentado
        ```

        ### Cómo correr todo en local

        ```bash
        git clone <url-del-repo>
        cd ShopSmart-Recommender
        python -m venv .venv && source .venv/bin/activate
        pip install -r requirements.txt

        # Reentrenar todo el pipeline (opcional, ya viene con artefactos entrenados):
        python -m src.pipeline.train_pipeline

        # Correr esta misma demo:
        streamlit run app.py
        ```
        """
    )
