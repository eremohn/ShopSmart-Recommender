"""Tests unitarios para las funciones de parsing en src/data/cleaning.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.cleaning import (
    extract_main_category,
    normalize_availability,
    parse_price,
    parse_rating_count,
    parse_rating_stars,
    parse_recent_purchases,
    parse_review_metadata,
)


def test_parse_price():
    assert parse_price("List Price: $53.99") == 53.99
    assert parse_price("Typical price: $1,234.50") == 1234.50
    assert pd.isna(parse_price(None))


def test_parse_rating_stars():
    assert parse_rating_stars("4.6 out of 5 stars") == 4.6
    assert pd.isna(parse_rating_stars(None))


def test_parse_rating_count():
    assert parse_rating_count("1,654 ratings") == 1654
    assert parse_rating_count("5 ratings") == 5


def test_parse_recent_purchases():
    assert parse_recent_purchases("50+ bought") == 50
    assert parse_recent_purchases("2K+ bought") == 2000


def test_parse_review_metadata():
    result = parse_review_metadata("Reviewed in the United States on March 6, 2025")
    assert result["review_country"] == "United States"
    assert str(result["review_date"].date()) == "2025-03-06"


def test_extract_main_category():
    assert extract_main_category("Clothing, Shoes & Jewelry › Men › Active") == "Men"


def test_normalize_availability():
    assert normalize_availability("In Stock") == "In Stock"
    assert normalize_availability("Only 1 left in stock - order soon.") == "Low Stock"
    assert normalize_availability("Currently unavailable.") == "Unavailable"
