"""Scrapy item containers.

Spiders fill these with *raw* strings straight from the page. ``DataCleaningPipeline``
validates them against the Pydantic models in ``models.py`` and writes the cleaned values
back, so every field here mirrors a model field (enforced by ``tests/test_validation.py``).
"""
import scrapy


class ProductItem(scrapy.Item):
    sku_id = scrapy.Field()
    title = scrapy.Field()
    platform = scrapy.Field()
    url = scrapy.Field()
    category = scrapy.Field()
    subcategory = scrapy.Field()
    brand = scrapy.Field()
    description = scrapy.Field()
    price = scrapy.Field()
    original_price = scrapy.Field()
    currency = scrapy.Field()
    rating = scrapy.Field()
    review_count = scrapy.Field()
    in_stock = scrapy.Field()
    stock_quantity = scrapy.Field()
    scraped_at = scrapy.Field()
    # Computed by ProductModel during validation.
    discount_pct = scrapy.Field()


class ReviewItem(scrapy.Item):
    review_id = scrapy.Field()
    sku_id = scrapy.Field()
    platform = scrapy.Field()
    author = scrapy.Field()
    rating = scrapy.Field()
    review_text = scrapy.Field()
    verified_purchase = scrapy.Field()
    date = scrapy.Field()
    scraped_at = scrapy.Field()
