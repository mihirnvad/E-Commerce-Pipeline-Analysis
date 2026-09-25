"""Metric functions and the synthetic generator (no database needed)."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from analysis.eda_analysis import (
    add_sentiment,
    discount_summary,
    popularity_summary,
    pricing_summary,
    sentiment_summary,
    trend_summary,
)
from scraper_engine.models import ProductModel, ReviewModel
from scripts import seed_50k_mock as seed_module
from scripts.seed_50k_mock import charm_price, generate_catalog

NOW = datetime(2025, 9, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def catalog():
    return generate_catalog(1_500, seed=11, now=NOW)


def test_generated_documents_satisfy_the_schemas(catalog):
    products, reviews = catalog
    assert len(products) == 1_500
    storage_only = {"price_history", "first_seen_at", "last_seen_at", "is_synthetic", "discount_pct"}
    for doc in products[:200]:
        ProductModel.model_validate({k: v for k, v in doc.items() if k not in storage_only})
    for doc in reviews[:200]:
        ReviewModel.model_validate({k: v for k, v in doc.items() if k != "is_synthetic"})
    assert all(doc["is_synthetic"] for doc in products + reviews)


def test_generated_keys_are_unique(catalog):
    products, reviews = catalog
    assert len({(p["platform"], p["sku_id"]) for p in products}) == len(products)
    assert len({(r["platform"], r["sku_id"], r["review_id"]) for r in reviews}) == len(reviews)


def test_generated_distributions_look_like_ecommerce(catalog):
    products, reviews = catalog
    df = pd.DataFrame(products)
    assert set(df["category"]) == {"Electronics", "Apparel", "Home Goods"}
    # Log-normal prices: long right tail, so the mean sits above the median.
    assert df["price"].mean() > df["price"].median()
    # Power-law popularity: the top 10% of listings hold most reviews.
    counts = np.sort(df["review_count"].to_numpy())[::-1]
    assert counts[: len(counts) // 10].sum() / counts.sum() > 0.5
    # Positively skewed stars.
    stars = pd.Series([r["rating"] for r in reviews])
    assert stars.mean() > 3.5 and (stars == 5).mean() > (stars == 2).mean()
    # Price history ends at today's price.
    assert all(p["price_history"][-1]["price"] == p["price"] for p in products)


def test_generation_is_deterministic_regardless_of_worker_count(monkeypatch):
    monkeypatch.setattr(seed_module, "CHUNK_SIZE", 40)  # force several chunks
    serial, serial_reviews = generate_catalog(100, seed=3, now=NOW, workers=1)
    parallel, parallel_reviews = generate_catalog(100, seed=3, now=NOW, workers=2)
    strip = lambda docs: [(d["sku_id"], d["price"], d["review_count"]) for d in docs]  # noqa: E731
    assert strip(serial) == strip(parallel)
    assert [r["review_id"] for r in serial_reviews] == [r["review_id"] for r in parallel_reviews]
    assert len({d["sku_id"] for d in serial}) == 100  # chunk offsets keep SKUs unique


@pytest.mark.parametrize(("raw", "expected"), [(7.2, 7.49), (7.8, 7.99), (24.3, 24.99), (847.0, 849.99)])
def test_charm_pricing(raw, expected):
    assert charm_price(raw) == expected


def frame():
    return pd.DataFrame({
        "category": ["A", "A", "A", "A", "B", "B"],
        "subcategory": ["x", "x", "y", "y", "z", "z"],
        "currency": ["USD"] * 5 + ["GBP"],
        "price": [10.0, 20.0, 30.0, 40.0, 100.0, 999.0],
        "discount_pct": [0.0, 20.0, 0.0, 10.0, 25.0, 50.0],
        "review_count": [0, 1, 2, 97, 0, 0],
        "rating": [0.0, 4.0, 4.5, 4.8, 0.0, 0.0],
    })


def test_pricing_summary_uses_only_the_analysis_currency():
    summary = pricing_summary(frame())
    assert summary.loc["A", "median"] == 25.0
    assert summary.loc["A", "iqr"] == pytest.approx(15.0)
    assert summary.loc["A", "share_discounted_pct"] == 50.0
    assert summary.loc["B", "listings"] == 1  # the GBP row is excluded
    assert list(summary.index) == ["B", "A"]  # ordered by median price


def test_discount_summary():
    stats = discount_summary(frame())
    assert stats["share_discounted_pct"] == pytest.approx(66.7)
    assert stats["median_depth_pct"] == 22.5
    assert stats["share_round_number_promos_pct"] == 100.0


def test_popularity_concentration():
    stats = popularity_summary(frame())
    assert stats["total_reviews"] == 100
    assert stats["top_10pct_share_of_reviews_pct"] == 97.0
    assert stats["top_subcategories"][0] == {"category": "A", "subcategory": "y", "reviews": 99}


class KeywordAnalyzer:
    """Stand-in for VADER so the test does not depend on the downloaded lexicon."""

    def polarity_scores(self, text):
        return {"compound": 0.8 if "love" in text else -0.6 if "hate" in text else 0.0}


def test_sentiment_summary_correlates_text_and_stars():
    reviews = pd.DataFrame({
        "rating": [5, 5, 4, 3, 2, 1, 1],
        "review_text": ["love it", "love", "love this", "meh", "hate", "hate it", "love that it broke"],
    })
    stats = sentiment_summary(add_sentiment(reviews, KeywordAnalyzer()))
    assert stats["pearson_r"] == pytest.approx(0.602, abs=0.001)  # dragged down by the sarcastic 1-star
    assert stats["mean_sentiment_by_star"][5] == 0.8
    assert stats["low_star_but_positive_text_pct"] == pytest.approx(33.3)  # the sarcastic one


def test_trend_summary():
    index = pd.DataFrame({
        "category": ["A", "A", "B"],
        "week": pd.to_datetime(["2025-01-06", "2025-01-13", "2025-01-06"]),
        "index": [100.0, 96.5, 100.0],
        "observations": [10, 10, 5],
    })
    assert trend_summary(index) == {"A": {"weeks": 2, "change_pct": -3.5}}
