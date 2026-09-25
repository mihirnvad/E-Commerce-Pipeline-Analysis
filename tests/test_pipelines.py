from datetime import datetime, timezone

import pytest
from pymongo import MongoClient
from scrapy import Spider
from scrapy.exceptions import DropItem
from scrapy.utils.test import get_crawler
from twisted.internet.task import Clock

from scraper_engine.items import ProductItem, ReviewItem
from scraper_engine.pipelines import PRICE_HISTORY_LIMIT, DataCleaningPipeline, MongoBatchUpsertPipeline
from scraper_engine.storage import PRODUCTS, REVIEWS
from tests.conftest import MONGO_TEST_URI


class DummySpider(Spider):
    name = "dummy"


def raw_product(sku="a897fe39b1053632", price="Â£51.77", **extra):
    fields = {
        "platform": "books_toscrape",
        "sku_id": sku,
        "title": "  A Light in the <em>Attic</em> ",
        "url": "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
        "category": "Home > Books > Poetry",
        "price": price,
        "rating": "star-rating Three",
        "review_count": "0",
        "in_stock": ["\n", "In stock (22 available)", "\n"],
    }
    fields.update(extra)
    return ProductItem(**fields)


def raw_review(review_id="r-1", **extra):
    fields = {
        "platform": "web_scraping_dev",
        "sku_id": "1",
        "review_id": review_id,
        "rating": 5,
        "review_text": "Absolutely delicious! The orange flavor is my favorite.",
        "date": "2022-07-22",
    }
    fields.update(extra)
    return ReviewItem(**fields)


# --------------------------------------------------------------------------- #
# DataCleaningPipeline
# --------------------------------------------------------------------------- #
@pytest.fixture
def cleaner():
    crawler = get_crawler(DummySpider)
    return DataCleaningPipeline.from_crawler(crawler)


def test_cleaning_pipeline_normalises_product(cleaner):
    item = cleaner.process_item(raw_product())
    assert isinstance(item, ProductItem)
    assert item["title"] == "A Light in the Attic"
    assert item["price"] == 51.77
    assert item["currency"] == "GBP"
    assert item["rating"] == 3.0
    assert item["in_stock"] is True
    assert item["stock_quantity"] == 22
    assert (item["category"], item["subcategory"]) == ("Books", "Poetry")
    assert item["discount_pct"] == 0.0
    assert cleaner.stats.get_value("cleaning/validated/ProductModel") == 1


def test_cleaning_pipeline_drops_invalid_items_and_counts_errors(cleaner):
    with pytest.raises(DropItem, match="price"):
        cleaner.process_item(raw_product(price="Sold out"))
    assert cleaner.stats.get_value("cleaning/dropped/ProductModel") == 1
    assert cleaner.stats.get_value("cleaning/errors/ProductModel/price") == 1


def test_cleaning_pipeline_normalises_review(cleaner):
    item = cleaner.process_item(raw_review(review_text="Great&nbsp;value!", author=""))
    assert isinstance(item, ReviewItem)
    assert item["review_text"] == "Great value!"
    assert item["author"] == "Anonymous"
    assert item["date"] == datetime(2022, 7, 22, tzinfo=timezone.utc)


def test_cleaning_pipeline_ignores_foreign_items(cleaner):
    assert cleaner.process_item({"anything": 1}) == {"anything": 1}


# --------------------------------------------------------------------------- #
# MongoBatchUpsertPipeline
# --------------------------------------------------------------------------- #
@pytest.fixture
def db(mongo_client, mongo_db_name):
    return mongo_client[mongo_db_name]


def make_pipeline(db, **kwargs):
    crawler = get_crawler(DummySpider)
    options = {"batch_size": 3, "flush_interval": 0, "stats": crawler.stats}
    options.update(kwargs)
    # Each pipeline owns (and closes) its client, like it does inside a real crawl.
    return MongoBatchUpsertPipeline(
        mongo_uri=MONGO_TEST_URI,
        mongo_db=db.name,
        client_factory=lambda uri: MongoClient(uri, tz_aware=True),
        **options,
    )


def clean(item):
    return DataCleaningPipeline().process_item(item)


def test_unique_indexes_are_created_on_open(db):
    pipeline = make_pipeline(db)
    pipeline.open_spider()
    products_index = db[PRODUCTS].index_information()["uniq_platform_sku"]
    assert products_index["unique"] is True
    assert products_index["key"] == [("platform", 1), ("sku_id", 1)]
    assert "uniq_platform_sku_review" in db[REVIEWS].index_information()


def test_buffer_flushes_when_batch_is_full(db):
    pipeline = make_pipeline(db)
    pipeline.open_spider()
    products = db[PRODUCTS]

    for i in range(2):
        pipeline.process_item(clean(raw_product(sku=f"sku-{i}")))
    assert products.count_documents({}) == 0  # still buffered

    pipeline.process_item(clean(raw_product(sku="sku-2")))
    assert products.count_documents({}) == 3
    assert pipeline.stats.get_value("mongo/products/flushes/batch_full") == 1


def test_remaining_items_flush_on_close(db):
    pipeline = make_pipeline(db, batch_size=500)
    pipeline.open_spider()
    pipeline.process_item(clean(raw_product()))
    pipeline.process_item(clean(raw_review()))
    pipeline.close_spider()
    assert db[PRODUCTS].count_documents({}) == 1
    assert db[REVIEWS].count_documents({}) == 1


def test_interval_flush_lands_data_from_a_slow_crawl(db):
    clock = Clock()
    pipeline = make_pipeline(db, batch_size=500, flush_interval=5.0, clock=clock)
    pipeline.open_spider()
    pipeline.process_item(clean(raw_product()))
    products = db[PRODUCTS]

    clock.advance(4.9)
    assert products.count_documents({}) == 0
    clock.advance(0.2)
    assert products.count_documents({}) == 1
    pipeline.close_spider()
    assert not pipeline._flusher.running


def test_rerunning_a_crawl_updates_instead_of_duplicating(db):
    for run, price in enumerate(("£51.77", "£45.00", "£47.50")):
        pipeline = make_pipeline(db)
        pipeline.open_spider()
        pipeline.process_item(clean(raw_product(price=price)))
        pipeline.process_item(clean(raw_product(sku="other-book")))
        pipeline.process_item(clean(raw_review()))
        pipeline.close_spider()

    products = db[PRODUCTS]
    assert products.count_documents({}) == 2
    assert db[REVIEWS].count_documents({}) == 1

    doc = products.find_one({"platform": "books_toscrape", "sku_id": "a897fe39b1053632"})
    assert doc["price"] == 47.5
    assert [obs["price"] for obs in doc["price_history"]] == [51.77, 45.0, 47.5]
    assert doc["first_seen_at"] <= doc["last_seen_at"]


def test_syndicated_review_is_kept_per_listing(db):
    # web-scraping.dev shows the same review ids on several product pages.
    pipeline = make_pipeline(db)
    pipeline.open_spider()
    pipeline.process_item(clean(raw_review(review_id="classic-leather-sneakers-1", sku_id="11")))
    pipeline.process_item(clean(raw_review(review_id="classic-leather-sneakers-1", sku_id="23")))
    pipeline.process_item(clean(raw_review(review_id="classic-leather-sneakers-1", sku_id="23")))
    pipeline.close_spider()
    assert sorted(doc["sku_id"] for doc in db[REVIEWS].find()) == ["11", "23"]


def test_price_history_is_capped(db):
    pipeline = make_pipeline(db, batch_size=1)
    pipeline.open_spider()
    for i in range(PRICE_HISTORY_LIMIT + 5):
        pipeline.process_item(clean(raw_product(price=f"£{10 + i}.00")))
    history = db[PRODUCTS].find_one()["price_history"]
    assert len(history) == PRICE_HISTORY_LIMIT
    assert history[-1]["price"] == 10 + PRICE_HISTORY_LIMIT + 4


def test_write_errors_are_counted_not_raised(db):
    pipeline = make_pipeline(db)
    pipeline.open_spider()

    class ExplodingCollection:
        def bulk_write(self, *_args, **_kwargs):
            from pymongo.errors import AutoReconnect

            raise AutoReconnect("mongo went away")

    pipeline.db = {PRODUCTS: ExplodingCollection(), REVIEWS: ExplodingCollection()}
    pipeline.process_item(clean(raw_product()))
    pipeline.flush_all()
    assert pipeline.stats.get_value("mongo/products/failed") == 1
