from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from scraper_engine.items import ProductItem, ReviewItem
from scraper_engine.models import (
    ProductModel,
    ReviewModel,
    clean_text,
    normalize_category,
    parse_availability,
    parse_date,
    parse_price,
    parse_rating,
)

NOW = datetime(2025, 9, 1, 12, 0, tzinfo=timezone.utc)


def product(**overrides):
    base = {
        "sku_id": "SKU-1",
        "title": "Wireless Headphones",
        "platform": "demo_shop",
        "url": "https://shop.example.com/p/1",
        "price": "$129.99",
        "rating": "4.5 out of 5 stars",
        "review_count": "1,024 ratings",
        "in_stock": "In stock",
        "category": "Home > Electronics > Headphones",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$1,299.99 USD", (1299.99, "USD")),
        ("£51.77", (51.77, "GBP")),
        ("Â£51.77", (51.77, "GBP")),  # UTF-8 read as latin-1
        ("1.299,99 €", (1299.99, "EUR")),
        ("EUR 12,50", (12.5, "EUR")),
        ("US$ 45", (45.0, "USD")),
        ("1,299", (1299.0, None)),
        ("Price: 9.99", (9.99, None)),
        (19.5, (19.5, None)),
        ("Call for price", (None, None)),
        (None, (None, None)),
    ],
)
def test_parse_price(raw, expected):
    assert parse_price(raw) == expected


def test_clean_text_repairs_entities_tags_and_whitespace():
    assert clean_text("  <b>Caf&eacute;</b>\n\tâ€™s  best mug​ ") == "Café ’s best mug"


def test_clean_text_keeps_legitimate_symbols():
    assert clean_text("Voltix™ Pro ½-inch — Crème") == "Voltix™ Pro ½-inch — Crème"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("star-rating Three", 3.0), ("4.5 out of 5 stars", 4.5), ("4,7", 4.7), (4, 4.0), ("", None), ("n/a", None)],
)
def test_parse_rating(raw, expected):
    assert parse_rating(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("In stock (22 available)", True),
        ("https://schema.org/InStock", True),
        ("https://schema.org/OutOfStock", False),
        ("Currently unavailable.", False),
        ("Sold Out", False),
        ("maybe", None),
    ],
)
def test_parse_availability(raw, expected):
    assert parse_availability(raw) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3 days ago", NOW - timedelta(days=3)),
        ("a month ago", NOW - timedelta(days=30)),
        ("2 weeks ago", NOW - timedelta(days=14)),
        ("yesterday", NOW - timedelta(days=1)),
        ("Reviewed in the United States on July 22, 2022", datetime(2022, 7, 22, tzinfo=timezone.utc)),
        ("2022-07-22", datetime(2022, 7, 22, tzinfo=timezone.utc)),
        ("2024-03-01T10:00:00+02:00", datetime(2024, 3, 1, 8, 0, tzinfo=timezone.utc)),
        (1_700_000_000, datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)),
    ],
)
def test_parse_date_handles_relative_and_absolute(raw, expected):
    assert parse_date(raw, now=NOW) == expected


def test_parse_date_rejects_garbage():
    with pytest.raises(ValueError):
        parse_date("not a date at all")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Home > Books > Poetry", ("Books", "Poetry")),
        ("computers > laptops", ("Electronics", "Laptops")),
        ("Phones / Touch", ("Electronics", "Touch")),
        ("apparel", ("Apparel", None)),
        ("consumables", ("Grocery", "Consumables")),
        ("Home > Kitchen > Cookware", ("Home Goods", "Cookware")),
        ("Electronics > iPhone Cases", ("Electronics", "iPhone Cases")),
        ("Pet Supplies", ("Other", "Pet Supplies")),
        ("", ("Other", None)),
    ],
)
def test_normalize_category(raw, expected):
    assert normalize_category(raw) == expected


# --------------------------------------------------------------------------- #
# ProductModel
# --------------------------------------------------------------------------- #
def test_product_model_cleans_dirty_listing():
    model = ProductModel.model_validate(product(price="$1,299.99 USD", original_price="$1,599.99"))
    assert model.price == 1299.99
    assert model.currency == "USD"
    assert model.original_price == 1599.99
    assert model.discount_pct == pytest.approx(18.75, abs=0.01)
    assert model.rating == 4.5
    assert model.review_count == 1024
    assert model.in_stock is True
    assert (model.category, model.subcategory) == ("Electronics", "Headphones")
    assert model.scraped_at.tzinfo is not None


def test_product_model_extracts_stock_quantity():
    model = ProductModel.model_validate(product(in_stock="In stock (22 available)"))
    assert model.in_stock is True
    assert model.stock_quantity == 22


def test_original_price_not_above_price_is_discarded():
    model = ProductModel.model_validate(product(price="20.00", currency="USD", original_price="19.00"))
    assert model.original_price is None
    assert model.discount_pct == 0.0


def test_dump_is_mongo_ready():
    doc = ProductModel.model_validate(product()).model_dump()
    assert isinstance(doc["url"], str)
    assert isinstance(doc["scraped_at"], datetime)
    assert "discount_pct" in doc


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"price": "-5"}, "price"),
        ({"price": "free"}, "price"),
        ({"price": 10.0}, "currency"),  # no currency anywhere
        ({"rating": "7 stars"}, "rating"),
        ({"url": "not-a-url"}, "url"),
        ({"review_count": -1}, "review_count"),
        ({"title": "   "}, "title"),
        ({"in_stock": "call us"}, "in_stock"),
        ({"unexpected": "field"}, "unexpected"),
    ],
)
def test_product_model_rejects_invalid_data(override, field):
    with pytest.raises(ValidationError) as exc_info:
        ProductModel.model_validate(product(**override))
    assert field in {str(error["loc"][0]) for error in exc_info.value.errors()}


# --------------------------------------------------------------------------- #
# ReviewModel
# --------------------------------------------------------------------------- #
def review(**overrides):
    base = {
        "review_id": 101,
        "sku_id": "SKU-1",
        "platform": "Demo Shop",
        "rating": "5.0 out of 5 stars",
        "review_text": "  Absolutely <br>love it! ",
        "verified_purchase": "Verified Purchase",
        "date": "2 weeks ago",
    }
    base.update(overrides)
    return base


def test_review_model_normalises_fields():
    model = ReviewModel.model_validate(review())
    assert model.review_id == "101"
    assert model.platform == "demo_shop"
    assert model.rating == 5
    assert model.review_text == "Absolutely love it!"
    assert model.verified_purchase is True
    assert model.author == "Anonymous"
    assert model.date < datetime.now(timezone.utc)


def test_review_author_from_schema_org_person():
    assert ReviewModel.model_validate(review(author={"@type": "Person", "name": "Dana"})).author == "Dana"


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"rating": 0}, "rating"),
        ({"rating": "great"}, "rating"),
        ({"review_text": ""}, "review_text"),
        ({"date": "2999-01-01"}, "date"),
    ],
)
def test_review_model_rejects_invalid_data(override, field):
    with pytest.raises(ValidationError) as exc_info:
        ReviewModel.model_validate(review(**override))
    assert field in {str(error["loc"][0]) for error in exc_info.value.errors()}


# --------------------------------------------------------------------------- #
# Items <-> models contract
# --------------------------------------------------------------------------- #
def test_item_fields_mirror_model_fields():
    assert set(ProductItem.fields) == set(ProductModel.model_fields) | set(ProductModel.model_computed_fields)
    assert set(ReviewItem.fields) == set(ReviewModel.model_fields)
