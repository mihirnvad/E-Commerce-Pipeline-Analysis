"""Create (idempotently) every index the pipeline and the analysis rely on.

    python scripts/setup_mongo.py                 # uses MONGO_URI / MONGO_DATABASE
    python scripts/setup_mongo.py --uri mongodb://localhost:27017 --db ecommerce

Indexes (defined in scraper_engine/storage.py):

products
    uniq_platform_sku        {platform: 1, sku_id: 1}   unique -> idempotent upserts
    category_price           {category: 1, price: 1}    category pages sorted by price
    rating_desc              {rating: -1}               top-rated listings
    title_text               {title: "text"}            keyword search
reviews
    uniq_platform_sku_review {platform: 1, sku_id: 1, review_id: 1} unique
    product_reviews_by_date  {platform: 1, sku_id: 1, date: -1}
    review_rating            {rating: 1}
"""
from __future__ import annotations

import argparse
import sys

try:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.setup_mongo)
except ImportError:  # python scripts/setup_mongo.py
    import _bootstrap  # noqa: F401

from pymongo.errors import OperationFailure, PyMongoError  # noqa: E402

from scraper_engine.storage import (  # noqa: E402
    PRODUCTS,
    REVIEWS,
    ensure_indexes,
    get_client,
    mongo_database_name,
    mongo_uri,
)


def describe(db) -> None:
    for name in (PRODUCTS, REVIEWS):
        collection = db[name]
        print(f"\n{db.name}.{name}: {collection.estimated_document_count():,} documents")
        for index_name, spec in collection.index_information().items():
            keys = ", ".join(f"{field}: {direction}" for field, direction in spec["key"])
            flags = " unique" if spec.get("unique") else ""
            print(f"  - {index_name:<26} {{{keys}}}{flags}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uri", default=mongo_uri(), help="MongoDB URI (default: $MONGO_URI)")
    parser.add_argument("--db", default=mongo_database_name(), help="database name (default: $MONGO_DATABASE)")
    args = parser.parse_args(argv)

    client = get_client(args.uri)
    try:
        client.admin.command("ping")
        db = client[args.db]
        ensure_indexes(db)
        print(f"Indexes ensured on {args.db}")
        describe(db)
    except OperationFailure as exc:
        print(f"Index creation failed: {exc}", file=sys.stderr)
        if exc.code == 11000:
            print("Duplicate (platform, sku_id) pairs already exist; dedupe before adding the unique index.",
                  file=sys.stderr)
        return 1
    except PyMongoError as exc:
        print(f"Could not reach MongoDB at {args.uri}: {exc}", file=sys.stderr)
        print("Start it with: docker compose up -d mongo", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
