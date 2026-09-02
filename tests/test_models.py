"""Tests unitarios para src/models/ (recomendadores y métricas)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.models.evaluation import (
    avg_recommendation_quality,
    catalog_coverage_at_k,
    category_precision_at_k,
    intra_list_diversity,
)
from src.models.recommenders import ContentBasedRecommender, PopularityRecommender


def _sample_features():
    # 4 productos en un espacio de 2 dimensiones: A y B muy parecidos,
    # C y D muy distintos de A/B y entre sí.
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    index = pd.DataFrame({"asin": ["A", "B", "C", "D"]})
    return matrix, index


def test_content_based_recommender_finds_closest_neighbor():
    matrix, index = _sample_features()
    model = ContentBasedRecommender(matrix, index)
    recs = model.recommend("A", k=1)
    assert recs.iloc[0]["asin"] == "B"


def test_content_based_recommender_excludes_query_item():
    matrix, index = _sample_features()
    model = ContentBasedRecommender(matrix, index)
    recs = model.recommend("A", k=3)
    assert "A" not in recs["asin"].values


def _sample_master():
    return pd.DataFrame(
        {
            "asin": ["A", "B", "C", "D"],
            "main_category": ["Men", "Men", "Women", "Women"],
            "bayesian_rating": [4.8, 4.5, 4.9, 4.0],
            "avg_sentiment": [0.5, 0.3, 0.6, 0.1],
            "n_reviews": [20, 5, 30, 1],
        }
    )


def test_popularity_recommender_ranks_by_score():
    model = PopularityRecommender(_sample_master())
    recs = model.recommend(k=4)
    # El producto con mejor rating, sentimiento y volumen debería
    # encabezar el ranking.
    assert recs.iloc[0]["asin"] == "C"


def test_popularity_recommender_filters_by_category():
    model = PopularityRecommender(_sample_master())
    recs = model.recommend(k=10, main_category="Men")
    assert set(recs["asin"]) == {"A", "B"}


def test_category_precision_at_k():
    product_index = pd.DataFrame(
        {"asin": ["A", "B", "C", "D"], "main_category": ["Men", "Men", "Women", "Women"]}
    )
    recs = {"A": ["B", "C"]}  # 1 de 2 coincide en categoría
    precision = category_precision_at_k(recs, product_index)
    assert precision == 0.5


def test_catalog_coverage_at_k():
    recs = {"A": ["B"], "C": ["B"]}
    coverage = catalog_coverage_at_k(recs, catalog_size=4)
    assert coverage == 0.25  # solo "B" fue recomendado, sobre 4 productos


def test_intra_list_diversity_higher_for_dissimilar_items():
    matrix, index = _sample_features()
    asin_to_pos = {a: i for i, a in enumerate(index["asin"])}

    similar_recs = {"query": ["A", "B"]}  # A y B son casi idénticos
    dissimilar_recs = {"query": ["A", "C"]}  # A y C son ortogonales

    div_similar = intra_list_diversity(similar_recs, matrix, asin_to_pos)
    div_dissimilar = intra_list_diversity(dissimilar_recs, matrix, asin_to_pos)
    assert div_dissimilar > div_similar


def test_avg_recommendation_quality():
    product_index = pd.DataFrame(
        {"asin": ["A", "B", "C", "D"], "bayesian_rating": [4.0, 5.0, 3.0, 2.0]}
    )
    recs = {"query": ["A", "B"]}
    quality = avg_recommendation_quality(recs, product_index)
    assert quality == 4.5


def test_category_precision_per_query():
    from src.models.evaluation import category_precision_per_query

    product_index = pd.DataFrame(
        {"asin": ["A", "B", "C", "D"], "main_category": ["Men", "Men", "Women", "Women"]}
    )
    recs = {"A": ["B", "C"], "C": ["D", "A"]}
    per_query = category_precision_per_query(recs, product_index)
    assert per_query["A"] == 0.5
    assert per_query["C"] == 0.5


def test_bootstrap_confidence_interval_contains_mean():
    from src.models.evaluation import bootstrap_confidence_interval

    values = [0.8, 0.9, 0.7, 0.8, 1.0, 0.6, 0.9]
    mean, lower, upper = bootstrap_confidence_interval(values, n_bootstrap=500)
    assert lower <= mean <= upper


def test_paired_wilcoxon_detects_no_difference():
    from src.models.evaluation import paired_wilcoxon_test

    values_a = {"x": 0.5, "y": 0.5, "z": 0.5}
    values_b = {"x": 0.5, "y": 0.5, "z": 0.5}
    result = paired_wilcoxon_test(values_a, values_b)
    assert result["p_value"] == 1.0
    assert result["mean_diff"] == 0.0


def test_paired_wilcoxon_detects_difference():
    from src.models.evaluation import paired_wilcoxon_test

    values_a = {f"q{i}": 0.9 for i in range(20)}
    values_b = {f"q{i}": 0.3 for i in range(20)}
    result = paired_wilcoxon_test(values_a, values_b)
    assert result["p_value"] < 0.05
    assert result["mean_diff"] > 0
