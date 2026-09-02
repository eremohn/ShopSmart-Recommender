"""Transformadores personalizados, compatibles con `sklearn.Pipeline`, para
la ingeniería de características del sistema de recomendación.

Todos los transformadores siguen la interfaz `fit`/`transform` de
scikit-learn (heredan de `BaseEstimator` y `TransformerMixin`) para poder
combinarse dentro de un `Pipeline`/`ColumnTransformer` de forma nativa,
incluyendo su serialización con `joblib`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RareCategoryBucketer(BaseEstimator, TransformerMixin):
    """Agrupa categorías poco frecuentes de una columna en una etiqueta
    ``"Otros"``, para evitar una explosión dimensional al aplicar
    One-Hot Encoding sobre variables categóricas de alta cardinalidad
    (por ejemplo, ``brand_name`` con 285 valores únicos).

    Parameters
    ----------
    column : str
        Columna de entrada sobre la que se detectan las categorías poco
        frecuentes.
    output_column : str
        Nombre de la nueva columna con las categorías ya agrupadas.
    min_freq : int
        Frecuencia mínima (cantidad de filas) que debe tener una categoría
        en el set de entrenamiento (`fit`) para conservarse tal cual.
    """

    def __init__(self, column: str, output_column: str, min_freq: int = 3):
        self.column = column
        self.output_column = output_column
        self.min_freq = min_freq

    def fit(self, X: pd.DataFrame, y=None) -> "RareCategoryBucketer":
        value_counts = X[self.column].value_counts()
        self.frequent_categories_ = set(value_counts[value_counts >= self.min_freq].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.output_column] = np.where(
            X[self.column].isin(self.frequent_categories_), X[self.column], "Otros"
        )
        return X


class TextCombiner(BaseEstimator, TransformerMixin):
    """Concatena varias columnas de texto en un único campo de texto libre,
    listo para vectorizar con `TfidfVectorizer`.

    Permite repetir columnas (parámetro `weights`) para dar más peso
    relativo a ciertos campos dentro del vector TF-IDF resultante sin
    necesidad de post-procesar la matriz esparsa (repetir un término en el
    corpus incrementa su frecuencia y, por lo tanto, su peso).

    Parameters
    ----------
    columns : list[str]
        Columnas de texto a combinar, en el orden en que se concatenan.
    output_column : str
        Nombre de la columna de salida con el texto combinado.
    weights : dict[str, int], optional
        Cantidad de veces que se repite cada columna. Por defecto, 1.
    """

    def __init__(self, columns: list[str], output_column: str, weights: dict | None = None):
        self.columns = columns
        self.output_column = output_column
        self.weights = weights or {}

    def fit(self, X: pd.DataFrame, y=None) -> "TextCombiner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        def combine_row(row: pd.Series) -> str:
            parts = []
            for col in self.columns:
                weight = self.weights.get(col, 1)
                value = str(row[col]) if pd.notna(row[col]) else ""
                parts.extend([value] * weight)
            return " ".join(parts).strip()

        X[self.output_column] = X[self.columns].apply(combine_row, axis=1)
        return X


class DataFrameColumnSelector(BaseEstimator, TransformerMixin):
    """Selecciona una única columna de un DataFrame y la devuelve como
    array 1D, formato requerido por `TfidfVectorizer` dentro de un
    `ColumnTransformer` cuando se prefiere ser explícito sobre la forma
    de entrada esperada.
    """

    def __init__(self, column: str):
        self.column = column

    def fit(self, X: pd.DataFrame, y=None) -> "DataFrameColumnSelector":
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].fillna("").values
