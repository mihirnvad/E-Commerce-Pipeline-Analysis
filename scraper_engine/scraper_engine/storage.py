"""MongoDB connection helpers and the canonical index definitions.

Shared by the Scrapy pipeline, ``scripts/setup_mongo.py``, the seeder and the EDA module
so that every entry point agrees on collection names, index names and connection defaults.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, TEXT, IndexModel, MongoClient
from pymongo.database import Database

load_dotenv()

DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_MONGO_DATABASE = "ecommerce"

PRODUCTS = "products"
REVIEWS = "reviews"

# Natural keys. Every write path upserts on these, which is what makes re-runs idempotent.
PRODUCT_KEY = ("platform", "sku_id")
# A review is keyed per listing, not globally: storefronts syndicate the same review
# (same id) across product variants, e.g. every colour of a shoe. Observed on
# web-scraping.dev, where all 56 review ids appear on more than one product page.
REVIEW_KEY = ("platform", "sku_id", "review_id")

UNIQUE_INDEXES: dict[str, list[IndexModel]] = {
    PRODUCTS: [
        IndexModel([(field, ASCENDING) for field in PRODUCT_KEY], name="uniq_platform_sku", unique=True),
    ],
    REVIEWS: [
        IndexModel([(field, ASCENDING) for field in REVIEW_KEY], name="uniq_platform_sku_review", unique=True),
    ],
}

SECONDARY_INDEXES: dict[str, list[IndexModel]] = {
    PRODUCTS: [
        # Category browse pages sorted/filtered by price.
        IndexModel([("category", ASCENDING), ("price", ASCENDING)], name="category_price"),
        # "Top rated" listings.
        IndexModel([("rating", DESCENDING)], name="rating_desc"),
        # Keyword search: db.products.find({"$text": {"$search": "wireless headphones"}}).
        IndexModel([("title", TEXT)], name="title_text", default_language="english"),
    ],
    REVIEWS: [
        # Newest reviews for one product.
        IndexModel(
            [("platform", ASCENDING), ("sku_id", ASCENDING), ("date", DESCENDING)],
            name="product_reviews_by_date",
        ),
        IndexModel([("rating", ASCENDING)], name="review_rating"),
    ],
}


def mongo_uri() -> str:
    return os.getenv("MONGO_URI", DEFAULT_MONGO_URI)


def mongo_database_name() -> str:
    return os.getenv("MONGO_DATABASE", DEFAULT_MONGO_DATABASE)


def get_client(uri: str | None = None, **kwargs) -> MongoClient:
    options = {"serverSelectionTimeoutMS": 5000, "appname": "ecommerce-pipeline", "tz_aware": True}
    options.update(kwargs)
    return MongoClient(uri or mongo_uri(), **options)


def get_database(uri: str | None = None, name: str | None = None, client: MongoClient | None = None) -> Database:
    client = client or get_client(uri)
    return client[name or mongo_database_name()]


def ensure_indexes(db: Database, *, unique_only: bool = False) -> dict[str, list[str]]:
    """Create indexes idempotently. Returns the index names per collection."""
    created: dict[str, list[str]] = {}
    for collection in (PRODUCTS, REVIEWS):
        models = list(UNIQUE_INDEXES[collection])
        if not unique_only:
            models += SECONDARY_INDEXES[collection]
        created[collection] = db[collection].create_indexes(models)
    return created
