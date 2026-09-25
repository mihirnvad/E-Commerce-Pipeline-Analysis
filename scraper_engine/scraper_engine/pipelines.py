"""Item pipelines: clean + validate, then batch-upsert into MongoDB."""
from __future__ import annotations

import logging
import time

from itemadapter import ItemAdapter
from pydantic import BaseModel, ValidationError
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError, PyMongoError
from scrapy.exceptions import DropItem

from scraper_engine.items import ProductItem, ReviewItem
from scraper_engine.models import ProductModel, ReviewModel, clean_text
from scraper_engine.storage import (
    DEFAULT_MONGO_DATABASE,
    DEFAULT_MONGO_URI,
    PRODUCTS,
    REVIEW_KEY,
    REVIEWS,
    ensure_indexes,
    get_client,
)

logger = logging.getLogger(__name__)

PRICE_HISTORY_LIMIT = 52  # keep roughly a year of weekly observations per product


class DataCleaningPipeline:
    """Normalises raw strings and validates every item against its Pydantic model.

    Invalid items are dropped with a per-field error counter in the crawl stats
    (``cleaning/errors/<Model>/<field>``), so data quality problems show up in the
    end-of-crawl stats dump instead of silently reaching the database.
    """

    MODELS: dict[type, type[BaseModel]] = {ProductItem: ProductModel, ReviewItem: ReviewModel}

    def __init__(self, stats=None):
        self.stats = stats

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.stats)

    @staticmethod
    def clean_value(value):
        if isinstance(value, str):
            return clean_text(value) or None
        if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            return clean_text(" ".join(value)) or None
        return value

    def process_item(self, item, spider=None):
        model_cls = self.MODELS.get(type(item))
        if model_cls is None:
            return item  # not ours to validate

        raw = {
            key: cleaned
            for key, value in ItemAdapter(item).items()
            if key in model_cls.model_fields and (cleaned := self.clean_value(value)) is not None
        }
        name = model_cls.__name__
        try:
            model = model_cls.model_validate(raw)
        except ValidationError as exc:
            for error in exc.errors():
                field = ".".join(str(part) for part in error["loc"]) or "__root__"
                self._inc(f"cleaning/errors/{name}/{field}")
            self._inc(f"cleaning/dropped/{name}")
            summary = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:3])
            raise DropItem(f"Invalid {name} ({raw.get('platform')}/{raw.get('sku_id')}): {summary}") from exc

        self._inc(f"cleaning/validated/{name}")
        return type(item)(**model.model_dump())

    def _inc(self, key: str) -> None:
        if self.stats is not None:
            self.stats.inc_value(key)


class MongoBatchUpsertPipeline:
    """Buffers ``UpdateOne(..., upsert=True)`` operations and flushes them with ``bulk_write``.

    * Idempotent: products are keyed on ``(platform, sku_id)`` and reviews on
      ``(platform, sku_id, review_id)``, both backed by unique indexes, so re-running a crawl
      updates documents in place instead of duplicating them.
    * Batched: a buffer is flushed when it reaches ``MONGO_BATCH_SIZE`` operations,
      every ``MONGO_FLUSH_INTERVAL`` seconds (so a slow crawl still lands data), and
      when the spider closes.
    * History-preserving: each product upsert appends to a capped ``price_history``
      array, which is what the pricing-trend analysis reads.
    """

    def __init__(
        self,
        mongo_uri: str = DEFAULT_MONGO_URI,
        mongo_db: str = DEFAULT_MONGO_DATABASE,
        batch_size: int = 500,
        flush_interval: float = 5.0,
        stats=None,
        client_factory=get_client,
        clock=None,
    ):
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.batch_size = max(1, batch_size)
        self.flush_interval = flush_interval
        self.stats = stats
        self.client_factory = client_factory
        self.clock = clock  # injectable twisted clock for tests
        self.client = None
        self.db = None
        self._buffers: dict[str, list[UpdateOne]] = {PRODUCTS: [], REVIEWS: []}
        self._flusher = None

    @classmethod
    def from_crawler(cls, crawler):
        settings = crawler.settings
        return cls(
            mongo_uri=settings.get("MONGO_URI", DEFAULT_MONGO_URI),
            mongo_db=settings.get("MONGO_DATABASE", DEFAULT_MONGO_DATABASE),
            batch_size=settings.getint("MONGO_BATCH_SIZE", 500),
            flush_interval=settings.getfloat("MONGO_FLUSH_INTERVAL", 5.0),
            stats=crawler.stats,
        )

    # -- lifecycle ----------------------------------------------------------- #
    def open_spider(self, spider=None):
        self.client = self.client_factory(self.mongo_uri)
        self.db = self.client[self.mongo_db]
        # The unique keys are what make upserts idempotent, so never run without them.
        ensure_indexes(self.db, unique_only=True)
        if self.flush_interval > 0:
            # Imported lazily: importing the reactor at module import time would
            # install the default reactor before Scrapy installs the asyncio one.
            from twisted.internet import task

            self._flusher = task.LoopingCall(self.flush_all, reason="interval")
            if self.clock is not None:
                self._flusher.clock = self.clock
            self._flusher.start(self.flush_interval, now=False)
        logger.info("Mongo pipeline connected to %s/%s (batch=%d)", self.mongo_uri, self.mongo_db, self.batch_size)

    def close_spider(self, spider=None):
        if self._flusher is not None and self._flusher.running:
            self._flusher.stop()
        self.flush_all(reason="close")
        if self.client is not None:
            self.client.close()

    # -- item handling ------------------------------------------------------- #
    def process_item(self, item, spider=None):
        if isinstance(item, ProductItem):
            collection, operation = PRODUCTS, self.product_upsert(ItemAdapter(item).asdict())
        elif isinstance(item, ReviewItem):
            collection, operation = REVIEWS, self.review_upsert(ItemAdapter(item).asdict())
        else:
            return item

        buffer = self._buffers[collection]
        buffer.append(operation)
        if len(buffer) >= self.batch_size:
            self.flush(collection, reason="batch_full")
        return item

    @staticmethod
    def product_upsert(doc: dict) -> UpdateOne:
        observed_at = doc["scraped_at"]
        observation = {"price": doc["price"], "original_price": doc.get("original_price"), "observed_at": observed_at}
        return UpdateOne(
            {"platform": doc["platform"], "sku_id": doc["sku_id"]},
            {
                "$set": {**doc, "last_seen_at": observed_at},
                "$setOnInsert": {"first_seen_at": observed_at},
                "$push": {"price_history": {"$each": [observation], "$slice": -PRICE_HISTORY_LIMIT}},
            },
            upsert=True,
        )

    @staticmethod
    def review_upsert(doc: dict) -> UpdateOne:
        key = {field: doc[field] for field in REVIEW_KEY}
        return UpdateOne(key, {"$set": doc}, upsert=True)

    # -- flushing ------------------------------------------------------------ #
    def flush_all(self, reason: str = "manual") -> None:
        for collection in self._buffers:
            self.flush(collection, reason=reason)

    def flush(self, collection: str, reason: str = "manual") -> None:
        operations = self._buffers[collection]
        if not operations or self.db is None:
            return
        self._buffers[collection] = []
        started = time.monotonic()
        try:
            result = self.db[collection].bulk_write(operations, ordered=False)
            self._record(collection, result.upserted_count, result.modified_count, 0)
        except BulkWriteError as exc:
            # ordered=False: everything except the failed operations was applied.
            details = exc.details
            errors = details.get("writeErrors", [])
            self._record(collection, details.get("nUpserted", 0), details.get("nModified", 0), len(errors))
            logger.error("Bulk write to %s had %d error(s); first: %s", collection, len(errors), errors[:1])
        except PyMongoError:
            self._record(collection, 0, 0, len(operations))
            logger.exception("Bulk write to %s failed; %d operations lost", collection, len(operations))
        logger.debug(
            "Flushed %d ops to %s in %.0f ms (%s)", len(operations), collection, (time.monotonic() - started) * 1000,
            reason,
        )
        if self.stats is not None:
            self.stats.inc_value(f"mongo/{collection}/flushes/{reason}")

    def _record(self, collection: str, upserted: int, modified: int, failed: int) -> None:
        if self.stats is None:
            return
        self.stats.inc_value(f"mongo/{collection}/upserted", upserted)
        self.stats.inc_value(f"mongo/{collection}/modified", modified)
        if failed:
            self.stats.inc_value(f"mongo/{collection}/failed", failed)
