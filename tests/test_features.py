"""Tests unitarios para src/features/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.features.build_features import aggregate_reviews_by_product, compute_bayesian_rating
from src.features.transformers import RareCategoryBucketer, TextCombiner


def _sample_reviews():
    return pd.DataFrame(
        {
            "productASIN": ["A1", "A1", "A2"],
            "reviewID": ["r1", "r2", "r3"],
            "rating": [5, 3, 1],
            "sentiment_score": [0.8, 0.2, -0.5],
            "verifiedPurchase": [True, False, True],
            "review_length_words": [50, 20, 10],
            "helpful_vote_log": [1.0, 0.0, 0.0],
            "cleaned_review_text": ["great product", "meh ok", "bad quality"],
        }
    )


def test_aggregate_reviews_by_product():
    result = aggregate_reviews_by_product(_sample_reviews())
    assert set(result["asin"]) == {"A1", "A2"}
    a1 = result.loc[result["asin"] == "A1"].iloc[0]
    assert a1["n_reviews"] == 2
    assert a1["avg_rating_reviews"] == 4.0
    assert "great product" in a1["voice_of_customer_text"]


def test_compute_bayesian_rating_shrinks_low_volume():
    df = pd.DataFrame(
        {"avg_rating_reviews": [5.0, 3.0], "n_reviews": [1, 100]}
    )
    wr = compute_bayesian_rating(df, min_votes_quantile=0.5)
    # El producto con 1 sola reseña de 5 estrellas debe "encogerse" hacia
    # la media global, quedando por debajo de 5.
    assert wr.iloc[0] < 5.0


def test_rare_category_bucketer():
    df = pd.DataFrame({"brand": ["A", "A", "A", "B", "C"]})
    bucketer = RareCategoryBucketer(column="brand", output_column="brand_b", min_freq=2)
    out = bucketer.fit_transform(df)
    assert out.loc[out["brand"] == "A", "brand_b"].unique().tolist() == ["A"]
    assert out.loc[out["brand"] == "B", "brand_b"].iloc[0] == "Otros"
    assert out.loc[out["brand"] == "C", "brand_b"].iloc[0] == "Otros"


def test_text_combiner_weights():
    df = pd.DataFrame({"title": ["Shirt"], "desc": ["Blue cotton shirt"]})
    combiner = TextCombiner(columns=["title", "desc"], output_column="text", weights={"title": 2})
    out = combiner.fit_transform(df)
    assert out["text"].iloc[0] == "Shirt Shirt Blue cotton shirt"
