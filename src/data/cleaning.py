"""Funciones de parsing y limpieza para los datasets crudos de Amazon.

Los datos fuente provienen de un scraping de Amazon, por lo que varios
campos numéricos llegan como texto libre con formatos heterogéneos
(por ejemplo, ``"4.6 out of 5 stars"`` o ``"1,654 ratings"``). Este módulo
concentra la lógica de parsing para que sea reutilizable entre notebooks
y testeable de forma unitaria.
"""

import re

import ftfy
import numpy as np
import pandas as pd

_PRICE_RE = re.compile(r"\$([\d,]+\.?\d*)")
_RATING_STARS_RE = re.compile(r"([\d.]+)\s+out of 5 stars")
_RATING_COUNT_RE = re.compile(r"([\d,]+)\s+rating")
_RECENT_PURCHASES_RE = re.compile(r"([\d.]+)(K)?\+?\s+bought", re.IGNORECASE)
_REVIEW_META_RE = re.compile(
    r"Reviewed in (?:the )?(?P<country>[A-Za-z .]+?) on (?P<date>[A-Za-z]+ \d{1,2}, \d{4})"
)


def fix_encoding(text: object) -> object:
    """Repara texto con mojibake (doble codificación UTF-8) usando ``ftfy``.

    Los campos de texto libre del scraping (reseñas, descripciones) llegan
    con emojis y tildes mal codificados, p. ej. ``"‚úçÔ∏è"`` en lugar de
    ``"✍️"``. Los valores nulos se devuelven sin modificar.
    """
    if pd.isna(text):
        return text
    return ftfy.fix_text(str(text))


def parse_price(text: object) -> float:
    """Extrae el valor numérico de campos como ``"List Price: $53.99"``."""
    if pd.isna(text):
        return np.nan
    match = _PRICE_RE.search(str(text))
    if not match:
        return np.nan
    return float(match.group(1).replace(",", ""))


def parse_rating_stars(text: object) -> float:
    """Extrae el rating numérico de ``"4.6 out of 5 stars"`` -> ``4.6``."""
    if pd.isna(text):
        return np.nan
    match = _RATING_STARS_RE.search(str(text))
    return float(match.group(1)) if match else np.nan


def parse_rating_count(text: object) -> float:
    """Extrae el número de ratings de ``"1,654 ratings"`` -> ``1654``."""
    if pd.isna(text):
        return np.nan
    match = _RATING_COUNT_RE.search(str(text))
    if not match:
        return np.nan
    return float(match.group(1).replace(",", ""))


def parse_recent_purchases(text: object) -> float:
    """Extrae compras recientes de ``"2K+ bought"`` -> ``2000``."""
    if pd.isna(text):
        return np.nan
    match = _RECENT_PURCHASES_RE.search(str(text))
    if not match:
        return np.nan
    value = float(match.group(1))
    if match.group(2):  # sufijo "K"
        value *= 1_000
    return value


def parse_review_metadata(text: object) -> pd.Series:
    """Separa ``reviewMetadata`` en país y fecha de la reseña.

    Ejemplo: ``"Reviewed in the United States on March 6, 2025"`` ->
    ``("United States", Timestamp("2025-03-06"))``.
    """
    if pd.isna(text):
        return pd.Series({"review_country": np.nan, "review_date": pd.NaT})
    match = _REVIEW_META_RE.search(str(text))
    if not match:
        return pd.Series({"review_country": np.nan, "review_date": pd.NaT})
    country = match.group("country").strip()
    date = pd.to_datetime(match.group("date"), format="%B %d, %Y", errors="coerce")
    return pd.Series({"review_country": country, "review_date": date})


def extract_main_category(breadcrumbs: object) -> object:
    """Obtiene la categoría principal desde ``breadcrumbs``.

    Ejemplo: ``"Clothing, Shoes & Jewelry › Men › Clothing › Active"``
    -> ``"Men"``.
    """
    if pd.isna(breadcrumbs):
        return np.nan
    parts = [p.strip() for p in str(breadcrumbs).split("›")]
    return parts[1] if len(parts) > 1 else np.nan


def normalize_availability(text: object) -> object:
    """Agrupa los estados de disponibilidad en categorías consistentes."""
    if pd.isna(text):
        return np.nan
    value = str(text).strip().lower()
    if value.startswith("in stock"):
        return "In Stock"
    if "currently unavailable" in value or "out of stock" in value:
        return "Unavailable"
    if "only" in value and "left in stock" in value:
        return "Low Stock"
    if "available to ship" in value:
        return "Ships in 1-2 days"
    if "will be released" in value:
        return "Pre-order"
    return "Other"


def clean_products(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aplica a `products.csv` crudo la misma secuencia de limpieza
    documentada y justificada en `notebooks/01_data_quality_eda.ipynb`
    (parsing de tipos, tratamiento de faltantes, winsorización), de forma
    reutilizable por el pipeline reproducible (`src/pipeline/train_pipeline.py`)."""
    df = df_raw.copy()

    df["price_value"] = df["price_value"].astype(float)
    df["list_price_value"] = df["list_price"].apply(parse_price)
    df["rating_stars_num"] = df["rating_stars"].apply(parse_rating_stars)
    df["rating_count_num"] = df["rating_count"].apply(parse_rating_count)
    df["recent_purchases_num"] = df["recent_purchases"].apply(parse_recent_purchases)
    df["availability_clean"] = df["availability"].apply(normalize_availability)
    df["main_category"] = df["breadcrumbs"].apply(extract_main_category)
    df["scrape_time"] = pd.to_datetime(
        df["scrape_time"], format="%m-%d-%Y %H:%M", errors="coerce"
    )
    for col in ["about_item", "product_description", "customer_review_summary", "title"]:
        df[col] = df[col].apply(fix_encoding)

    df["model_number"] = df["model_number"].fillna("No especificado")
    df["manufacturer"] = df["manufacturer"].fillna("No especificado")
    df["product_description"] = df["product_description"].fillna("")
    df["customer_review_summary"] = df["customer_review_summary"].fillna("")

    df["has_rank"] = df["rank_1"].notna()
    df["rank_1"] = df["rank_1"].fillna(df["rank_1"].max() + 1)
    df["has_recent_purchases"] = df["recent_purchases_num"].notna()
    df["recent_purchases_num"] = df["recent_purchases_num"].fillna(0)

    for col in ["price_value", "rating_count_num", "rating_stars_num"]:
        df[col] = df[col].fillna(df[col].median())
    df["list_price_value"] = df["list_price_value"].fillna(df["price_value"])

    df["main_category"] = df["main_category"].fillna("Sin categoría")
    df["availability_clean"] = df["availability_clean"].fillna("Other")

    p99_price = df["price_value"].quantile(0.99)
    df["price_value_winsorized"] = df["price_value"].clip(upper=p99_price)
    df["rating_count_log"] = np.log1p(df["rating_count_num"])
    df["recent_purchases_log"] = np.log1p(df["recent_purchases_num"])

    return df


def clean_reviews(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aplica a `reviews.csv` crudo la misma secuencia de limpieza
    documentada en `notebooks/01_data_quality_eda.ipynb`."""
    df = df_raw.copy()

    review_meta = df["reviewMetadata"].apply(parse_review_metadata)
    df = pd.concat([df, review_meta], axis=1)

    for col in ["reviewText", "reviewTitle", "cleaned_review_text"]:
        df[col] = df[col].apply(fix_encoding)

    df["review_length_words"] = df["reviewText"].fillna("").apply(lambda t: len(str(t).split()))

    media_cols = [c for c in df.columns if c.startswith(("images/", "videos/"))]
    df["has_media"] = df[media_cols].notna().any(axis=1)
    df = df.drop(columns=media_cols + ["reviewMetadata"])

    df["productVariant"] = df["productVariant"].fillna("Estándar")
    df = df.dropna(subset=["reviewText", "rating", "cleaned_review_text"])
    df["review_country"] = df["review_country"].fillna("Desconocido")
    df["reviewTitle"] = df["reviewTitle"].fillna("Sin título")
    df["helpful_vote_log"] = np.log1p(df["helpfulVoteCount"])

    return df
