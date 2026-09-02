"""Modelos de recomendación del MVP de ShopSmart Recommender.

Dado que `reviews.csv` no incluye un identificador de usuario/comprador
(ver `notebooks/02_feature_engineering.ipynb`), este proyecto no
implementa un filtrado colaborativo basado en usuarios. En su lugar,
compara dos enfoques **no-personalizado / personalizado por producto**,
un patrón estándar en sistemas de recomendación:

- `ContentBasedRecommender`: recomendación **personalizada por producto**
  (similitud coseno sobre la matriz de features de
  `02_feature_engineering.ipynb`). Responde a "si te gusta este producto,
  también te puede interesar...".
- `PopularityRecommender`: recomendación **no personalizada** ("Top
  Picks" / "Trending Now"), basada en un score compuesto de rating
  bayesiano, sentimiento y volumen de reseñas. Es el baseline estándar
  contra el que se compara cualquier recomendador personalizado, y la
  aproximación de este proyecto al componente "colaborativo-implícito".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """Recomendador basado en similitud de contenido (coseno).

    Parameters
    ----------
    features_matrix : np.ndarray
        Matriz densa de features por producto (salida de
        `feature_engineering_pipeline`), shape (n_productos, n_dims).
    product_index : pd.DataFrame
        DataFrame con al menos la columna `asin`, en el mismo orden de
        filas que `features_matrix`.
    """

    def __init__(self, features_matrix: np.ndarray, product_index: pd.DataFrame):
        self.features_matrix = features_matrix
        self.product_index = product_index.reset_index(drop=True)
        self._asin_to_pos = {asin: pos for pos, asin in enumerate(self.product_index["asin"])}
        self._similarity_matrix: np.ndarray | None = None

    def _similarity(self) -> np.ndarray:
        """Calcula (una única vez, de forma perezosa) la matriz de
        similitud coseno completa entre todos los productos."""
        if self._similarity_matrix is None:
            self._similarity_matrix = cosine_similarity(self.features_matrix)
        return self._similarity_matrix

    def recommend(self, asin: str, k: int = 10) -> pd.DataFrame:
        """Devuelve los `k` productos más similares al `asin` dado.

        Returns
        -------
        pd.DataFrame
            Columnas: `asin`, `similarity_score`, ordenado descendente.
        """
        if asin not in self._asin_to_pos:
            raise KeyError(f"ASIN '{asin}' no encontrado en el índice de productos.")

        pos = self._asin_to_pos[asin]
        scores = self._similarity()[pos]

        top_positions = np.argsort(scores)[::-1]
        top_positions = top_positions[top_positions != pos][:k]

        result = self.product_index.iloc[top_positions][["asin"]].copy()
        result["similarity_score"] = scores[top_positions]
        return result.reset_index(drop=True)


class PopularityRecommender:
    """Recomendador no-personalizado ("Top Picks"), basado en un score
    compuesto de calidad percibida.

    ``score = w_rating * bayesian_rating_norm + w_sentiment *
    avg_sentiment_norm + w_volume * n_reviews_norm``

    Las tres señales se normalizan a [0, 1] (min-max) antes de combinarse,
    para que ninguna domine por su escala original.
    """

    def __init__(
        self,
        products_master: pd.DataFrame,
        w_rating: float = 0.5,
        w_sentiment: float = 0.3,
        w_volume: float = 0.2,
    ):
        self.products_master = products_master.reset_index(drop=True).copy()
        self.w_rating = w_rating
        self.w_sentiment = w_sentiment
        self.w_volume = w_volume
        self._fit()

    @staticmethod
    def _min_max_norm(series: pd.Series) -> pd.Series:
        span = series.max() - series.min()
        if span == 0:
            return pd.Series(0.5, index=series.index)
        return (series - series.min()) / span

    def _fit(self) -> None:
        df = self.products_master
        rating_norm = self._min_max_norm(df["bayesian_rating"])
        sentiment_norm = self._min_max_norm(df["avg_sentiment"])
        volume_norm = self._min_max_norm(np.log1p(df["n_reviews"]))

        df["popularity_score"] = (
            self.w_rating * rating_norm
            + self.w_sentiment * sentiment_norm
            + self.w_volume * volume_norm
        )
        self.products_master = df.sort_values("popularity_score", ascending=False).reset_index(
            drop=True
        )

    def recommend(self, k: int = 10, main_category: str | None = None) -> pd.DataFrame:
        """Devuelve el top-`k` global (o filtrado por categoría) según
        `popularity_score`.

        Parameters
        ----------
        main_category : str, optional
            Si se especifica, restringe el ranking a una categoría
            ("Trending en Hombres", por ejemplo). Si es `None`, devuelve
            el ranking global (mismo resultado para cualquier usuario:
            comportamiento no-personalizado por diseño).
        """
        df = self.products_master
        if main_category is not None:
            df = df[df["main_category"] == main_category]
        return df.head(k)[["asin", "popularity_score"]].reset_index(drop=True)
