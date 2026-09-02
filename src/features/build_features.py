"""Agregación de reseñas a nivel producto.

`reviews.csv` no contiene un identificador de usuario/comprador, solo
`reviewID` (identificador de la reseña) y `productASIN`. Por lo tanto, no
es posible construir una matriz usuario-ítem tradicional para filtrado
colaborativo basado en usuarios.

Como alternativa, este módulo agrega la "sabiduría de las multitudes"
contenida en las reseñas (rating, sentimiento, votos útiles) a nivel de
producto, generando una señal de tipo colaborativo-implícito que
complementa a las características de contenido del producto.
"""

import numpy as np
import pandas as pd


def aggregate_reviews_by_product(reviews: pd.DataFrame) -> pd.DataFrame:
    """Agrega el dataset de reseñas a nivel de producto (`productASIN`).

    Parameters
    ----------
    reviews : pd.DataFrame
        Dataset de reseñas limpio (salida del notebook 01), debe contener
        las columnas: `productASIN`, `rating`, `sentiment_score`,
        `verifiedPurchase`, `review_length_words`, `helpful_vote_log`,
        `cleaned_review_text`.

    Returns
    -------
    pd.DataFrame
        Un registro por producto con features agregadas de sus reseñas.
    """
    grouped = reviews.groupby("productASIN")

    agg = grouped.agg(
        n_reviews=("reviewID", "count"),
        avg_rating_reviews=("rating", "mean"),
        std_rating_reviews=("rating", "std"),
        avg_sentiment=("sentiment_score", "mean"),
        std_sentiment=("sentiment_score", "std"),
        avg_review_length=("review_length_words", "mean"),
        avg_helpful_votes_log=("helpful_vote_log", "mean"),
        pct_verified=("verifiedPurchase", "mean"),
    )

    agg["pct_positive_reviews"] = grouped.apply(
        lambda g: (g["rating"] >= 4).mean(), include_groups=False
    )
    agg["pct_negative_reviews"] = grouped.apply(
        lambda g: (g["rating"] <= 2).mean(), include_groups=False
    )

    # Texto concatenado de todas las reseñas del producto: la "voz del
    # cliente" como feature textual adicional para el modelo de contenido.
    agg["voice_of_customer_text"] = grouped["cleaned_review_text"].apply(
        lambda texts: " ".join(texts.dropna())
    )

    # Desviación estándar indefinida (un solo review) -> se completa con 0
    # (no hay variabilidad observable, no es un dato faltante real).
    agg["std_rating_reviews"] = agg["std_rating_reviews"].fillna(0)
    agg["std_sentiment"] = agg["std_sentiment"].fillna(0)

    agg = agg.reset_index().rename(columns={"productASIN": "asin"})
    return agg


def compute_bayesian_rating(
    df: pd.DataFrame,
    rating_col: str = "avg_rating_reviews",
    count_col: str = "n_reviews",
    min_votes_quantile: float = 0.5,
) -> pd.Series:
    """Calcula un rating ponderado bayesiano (fórmula estilo IMDb).

    Corrige el sesgo de productos con pocas reseñas pero rating perfecto
    (ej. 1 sola reseña de 5 estrellas) descontando su calificación hacia
    la media global, en proporción inversa a su cantidad de reseñas.

    ``WR = (v / (v + m)) * R + (m / (v + m)) * C``

    donde ``R`` es el rating promedio del producto, ``v`` su cantidad de
    reseñas, ``C`` el rating promedio global y ``m`` la cantidad mínima de
    reseñas (percentil `min_votes_quantile`) para que el rating "pese"
    por sí solo.
    """
    global_mean = df[rating_col].mean()
    min_votes = df[count_col].quantile(min_votes_quantile)

    v = df[count_col]
    r = df[rating_col]
    return (v / (v + min_votes)) * r + (min_votes / (v + min_votes)) * global_mean
