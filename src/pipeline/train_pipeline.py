"""Pipeline reproducible end-to-end de ShopSmart Recommender.

Encadena los pasos documentados en los notebooks 01-04
(limpieza de datos -> ingeniería de características -> entrenamiento de
modelos -> evaluación) en un único script ejecutable, registrando
parámetros, métricas y artefactos en MLflow para que cada ejecución sea
comparable contra el baseline documentado en
`docs/VALIDATION_PLAN.md`.

Uso
----
```bash
python -m src.pipeline.train_pipeline
python -m src.pipeline.train_pipeline --n-components 150 --w-rating 0.6
mlflow ui --backend-store-uri sqlite:///mlflow.db  # para explorar los runs registrados
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaning import clean_products, clean_reviews  # noqa: E402
from src.data.load_data import load_products, load_reviews  # noqa: E402
from src.features.build_features import (  # noqa: E402
    aggregate_reviews_by_product,
    compute_bayesian_rating,
)
from src.features.transformers import (  # noqa: E402
    DataFrameColumnSelector,
    RareCategoryBucketer,
    TextCombiner,
)
from src.models.evaluation import (  # noqa: E402
    avg_recommendation_quality,
    catalog_coverage_at_k,
    category_precision_at_k,
    intra_list_diversity,
)
from src.models.recommenders import ContentBasedRecommender, PopularityRecommender  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

NUMERIC_FEATURES = [
    "price_value_winsorized",
    "rating_stars_num",
    "rating_count_log",
    "recent_purchases_log",
    "n_reviews",
    "bayesian_rating",
    "avg_sentiment",
    "std_sentiment",
    "pct_positive_reviews",
    "pct_negative_reviews",
    "pct_verified",
    "avg_helpful_votes_log",
]
CATEGORICAL_FEATURES = ["main_category", "availability_clean", "brand_name_bucketed"]


def build_master_table(random_seed: int = 42) -> pd.DataFrame:
    """Carga y limpia `products.csv`/`reviews.csv`, agrega las reseñas a
    nivel producto y calcula el rating bayesiano. Equivale a los pasos 1-4
    de `notebooks/02_feature_engineering.ipynb`."""
    df_products = clean_products(load_products())
    df_reviews = clean_reviews(load_reviews())

    df_review_agg = aggregate_reviews_by_product(df_reviews)
    df_master = df_products.merge(df_review_agg, on="asin", how="left")

    df_master["has_reviews"] = df_master["n_reviews"].notna()
    for col in ["n_reviews", "pct_verified", "pct_positive_reviews", "pct_negative_reviews"]:
        df_master[col] = df_master[col].fillna(0)
    for col in ["avg_rating_reviews", "avg_sentiment", "avg_review_length", "avg_helpful_votes_log"]:
        df_master[col] = df_master[col].fillna(df_master[col].median())
    df_master["std_rating_reviews"] = df_master["std_rating_reviews"].fillna(0)
    df_master["std_sentiment"] = df_master["std_sentiment"].fillna(0)
    df_master["voice_of_customer_text"] = df_master["voice_of_customer_text"].fillna("")

    df_master["bayesian_rating"] = compute_bayesian_rating(
        df_master, rating_col="avg_rating_reviews", count_col="n_reviews"
    )
    return df_master


def build_feature_pipeline(brand_min_freq: int, n_components: int, random_seed: int) -> Pipeline:
    """Construye (sin entrenar) el pipeline de ingeniería de
    características, idéntico en estructura al del notebook 02."""
    preprocessing_steps = Pipeline(
        steps=[
            (
                "bucket_brand",
                RareCategoryBucketer(
                    column="brand_name",
                    output_column="brand_name_bucketed",
                    min_freq=brand_min_freq,
                ),
            ),
            (
                "combine_content_text",
                TextCombiner(
                    columns=[
                        "title",
                        "main_category",
                        "brand_name",
                        "about_item",
                        "product_description",
                    ],
                    output_column="content_text",
                    weights={"title": 2},
                ),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            (
                "content_text",
                Pipeline(
                    steps=[
                        ("select", DataFrameColumnSelector(column="content_text")),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                max_features=400, ngram_range=(1, 2), stop_words="english", min_df=2
                            ),
                        ),
                    ]
                ),
                ["content_text"],
            ),
            (
                "voice_of_customer",
                Pipeline(
                    steps=[
                        ("select", DataFrameColumnSelector(column="voice_of_customer_text")),
                        ("tfidf", TfidfVectorizer(max_features=200, ngram_range=(1, 1), min_df=2)),
                    ]
                ),
                ["voice_of_customer_text"],
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing_steps),
            ("features", column_transformer),
            ("reduce_dim", TruncatedSVD(n_components=n_components, random_state=random_seed)),
        ]
    )


def evaluate_models(
    content_model: ContentBasedRecommender,
    popularity_model: PopularityRecommender,
    product_index: pd.DataFrame,
    features_matrix: np.ndarray,
    asin_to_pos: dict,
    k: int = 10,
) -> pd.DataFrame:
    """Reproduce la evaluación offline del notebook 03: 4 métricas para
    el modelo content-based y para el modelo de popularidad (global)."""
    all_asins = product_index["asin"].tolist()

    content_recs = {a: content_model.recommend(a, k=k)["asin"].tolist() for a in all_asins}
    popularity_recs = {a: popularity_model.recommend(k=k)["asin"].tolist() for a in all_asins}

    rows = []
    for name, recs in [("content_based", content_recs), ("popularity", popularity_recs)]:
        rows.append(
            {
                "modelo": name,
                "category_precision_at_10": category_precision_at_k(recs, product_index),
                "catalog_coverage_at_10": catalog_coverage_at_k(recs, len(all_asins)),
                "intra_list_diversity_at_10": intra_list_diversity(
                    recs, features_matrix, asin_to_pos
                ),
                "avg_quality_at_10": avg_recommendation_quality(recs, product_index.merge(
                    popularity_model.products_master[["asin", "bayesian_rating"]], on="asin"
                )),
            }
        )
    return pd.DataFrame(rows).set_index("modelo")


def run_pipeline(
    brand_min_freq: int = 3,
    n_components: int = 100,
    w_rating: float = 0.5,
    w_sentiment: float = 0.3,
    w_volume: float = 0.2,
    random_seed: int = 42,
    experiment_name: str = "shopsmart-recommender",
    models_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> dict:
    """Ejecuta el pipeline completo y registra todo en MLflow.

    Parameters
    ----------
    models_dir, processed_dir : Path, optional
        Directorios de salida para los artefactos entrenados. Por
        defecto usan `models/` y `data/processed/` del proyecto — se
        pueden sobreescribir (p. ej. con un directorio temporal) para
        que un test de integración no pise los artefactos de producción
        que consume la demo de Streamlit.

    Returns
    -------
    dict
        Resumen con las rutas de los artefactos generados y el DataFrame
        de métricas de evaluación, útil para tests/CI.
    """
    models_dir = models_dir or MODELS_DIR
    processed_dir = processed_dir or PROCESSED_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    # MLflow >= 3.x deprecó el backend de archivos plano ("./mlruns");
    # se usa SQLite como backend de tracking, tal como recomienda la
    # documentación oficial para proyectos nuevos.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "brand_min_freq": brand_min_freq,
                "n_components": n_components,
                "w_rating": w_rating,
                "w_sentiment": w_sentiment,
                "w_volume": w_volume,
                "random_seed": random_seed,
            }
        )

        df_master = build_master_table(random_seed=random_seed)

        feature_pipeline = build_feature_pipeline(brand_min_freq, n_components, random_seed)
        X_features = feature_pipeline.fit_transform(df_master)

        product_index = df_master[["asin", "title", "main_category"]].reset_index(drop=True)
        asin_to_pos = {a: i for i, a in enumerate(product_index["asin"])}

        content_model = ContentBasedRecommender(X_features, product_index)
        popularity_model = PopularityRecommender(
            df_master, w_rating=w_rating, w_sentiment=w_sentiment, w_volume=w_volume
        )

        df_evaluation = evaluate_models(
            content_model, popularity_model, product_index, X_features, asin_to_pos
        )

        for model_name, row in df_evaluation.iterrows():
            for metric_name, value in row.items():
                mlflow.log_metric(f"{model_name}__{metric_name}", value)

        joblib.dump(feature_pipeline, models_dir / "feature_engineering_pipeline.joblib")
        joblib.dump(content_model, models_dir / "content_based_recommender.joblib")
        joblib.dump(popularity_model, models_dir / "popularity_recommender.joblib")

        np.save(processed_dir / "product_features_matrix.npy", X_features)
        product_index.to_csv(processed_dir / "product_features_index.csv", index=False)
        df_master.to_csv(processed_dir / "products_master.csv", index=False)
        df_evaluation.to_csv(processed_dir / "model_evaluation_results.csv")

        for artifact in [
            models_dir / "feature_engineering_pipeline.joblib",
            models_dir / "content_based_recommender.joblib",
            models_dir / "popularity_recommender.joblib",
            processed_dir / "model_evaluation_results.csv",
        ]:
            mlflow.log_artifact(str(artifact))

        run_id = mlflow.active_run().info.run_id

    return {
        "run_id": run_id,
        "evaluation": df_evaluation,
        "n_products": len(df_master),
        "feature_matrix_shape": X_features.shape,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline reproducible de ShopSmart Recommender")
    parser.add_argument("--brand-min-freq", type=int, default=3)
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--w-rating", type=float, default=0.5)
    parser.add_argument("--w-sentiment", type=float, default=0.3)
    parser.add_argument("--w-volume", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--experiment-name", type=str, default="shopsmart-recommender")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_pipeline(
        brand_min_freq=args.brand_min_freq,
        n_components=args.n_components,
        w_rating=args.w_rating,
        w_sentiment=args.w_sentiment,
        w_volume=args.w_volume,
        random_seed=args.random_seed,
        experiment_name=args.experiment_name,
    )
    print(f"Run MLflow: {result['run_id']}")
    print(f"Productos procesados: {result['n_products']}")
    print(f"Matriz de features: {result['feature_matrix_shape']}")
    print("\nEvaluación:")
    print(result["evaluation"].round(4))
