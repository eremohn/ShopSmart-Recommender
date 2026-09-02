"""Funciones de carga de datos crudos para el proyecto ShopSmart Recommender.

Este módulo centraliza la lectura de los archivos fuente (``products.csv`` y
``reviews.csv``) para evitar duplicar rutas y parámetros de lectura a lo
largo de distintos notebooks y scripts.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTS_PATH = PROJECT_ROOT / "data" / "raw" / "products" / "products.csv"
REVIEWS_PATH = PROJECT_ROOT / "data" / "raw" / "reviews" / "reviews.csv"


def load_products(path: Path = PRODUCTS_PATH) -> pd.DataFrame:
    """Carga el dataset crudo de productos.

    Parameters
    ----------
    path : Path
        Ruta al archivo ``products.csv``.

    Returns
    -------
    pd.DataFrame
        Dataset de productos sin transformar.
    """
    return pd.read_csv(path, encoding="utf-8")


def load_reviews(path: Path = REVIEWS_PATH) -> pd.DataFrame:
    """Carga el dataset crudo de reseñas.

    Parameters
    ----------
    path : Path
        Ruta al archivo ``reviews.csv``.

    Returns
    -------
    pd.DataFrame
        Dataset de reseñas sin transformar.
    """
    return pd.read_csv(path, encoding="utf-8")
