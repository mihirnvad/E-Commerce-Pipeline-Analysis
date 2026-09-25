"""Multi-platform e-commerce product spider.

Targets three public sandboxes that exist specifically for scraping practice, each
chosen because it exercises a different extraction technique:

``books_toscrape``   (books.toscrape.com)
    Classic server-rendered catalogue: 50 paginated listing pages -> 1,000 detail pages.
``webscraper_io``    (webscraper.io/test-sites/e-commerce)
    The listing is rendered client-side from a JSON payload embedded in a ``data-items``
    attribute. By default the spider reads that payload directly, with no browser. With
    ``ENABLE_PLAYWRIGHT=1`` it instead renders the infinite-scroll variant in headless
    Chromium and scrolls until the product grid stops growing.
``web_scraping_dev`` (web-scraping.dev)
    Reviews are injected by JavaScript from a hidden ``<script id="reviews-data">`` JSON
    blob, and prices/ratings are also published as schema.org JSON-LD. The spider reads
    both, so it gets the dynamic content without rendering.

Usage (from the ``scraper_engine/`` directory):

    scrapy crawl ecommerce                                   # all platforms
    scrapy crawl ecommerce -a platforms=books_toscrape -a max_pages=2
    ENABLE_PLAYWRIGHT=1 scrapy crawl ecommerce -a platforms=webscraper_io
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse

import scrapy
from scrapy.http import Response

from scraper_engine.items import ProductItem, ReviewItem

try:  # optional dependency, only needed for ENABLE_PLAYWRIGHT=1
    from scrapy_playwright.page import PageMethod
except ImportError:  # pragma: no cover - exercised only when the extra is missing
    PageMethod = None

# Scrolls until the page height has been stable for 3 consecutive checks (max 80 rounds).
# One unchanged check is not enough: lazy loaders often need a beat before appending.
# Verified to load all 117/117 cards on webscraper.io's infinite-scroll laptops page.
INFINITE_SCROLL_JS = """
async () => {
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  let lastHeight = 0;
  let stableRounds = 0;
  for (let round = 0; round < 80 && stableRounds < 3; round++) {
    window.scrollTo(0, document.documentElement.scrollHeight);
    window.dispatchEvent(new Event("scroll"));
    await sleep(500);
    const height = document.documentElement.scrollHeight;
    stableRounds = height === lastHeight ? stableRounds + 1 : 0;
    lastHeight = height;
  }
}
"""

WEBSCRAPER_BASE = "https://webscraper.io/test-sites/e-commerce"
WEBSCRAPER_CATEGORIES = ("computers/laptops", "computers/tablets", "phones/touch")
WEB_SCRAPING_DEV_CATEGORIES = ("apparel", "consumables", "household")
# Keep AutoThrottle from lowering the robots.txt Crawl-delay configured in DOWNLOAD_SLOTS.
WSD_META = {"autothrottle_dont_adjust_delay": True}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def iter_json_ld(response: Response) -> Iterator[dict]:
    """Yield every JSON-LD object on the page, flattening lists and ``@graph`` containers."""
    for raw in response.css('script[type="application/ld+json"]::text').getall():
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            node = stack.pop(0)
            if not isinstance(node, dict):
                continue
            if "@graph" in node:
                stack.extend(node["@graph"])
            yield node


def find_json_ld(response: Response, schema_type: str) -> dict:
    for node in iter_json_ld(response):
        node_type = node.get("@type")
        types = node_type if isinstance(node_type, list) else [node_type]
        if schema_type in types:
            return node
    return {}


def stable_id(*parts: Any) -> str:
    """Deterministic id for records the site does not give an id to (e.g. JSON-LD reviews)."""
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


class EcommerceSpider(scrapy.Spider):
    name = "ecommerce"
    allowed_domains = ["books.toscrape.com", "webscraper.io", "web-scraping.dev"]
    PLATFORMS = ("books_toscrape", "webscraper_io", "web_scraping_dev")

    def __init__(self, platforms: str = "all", max_pages: str | int | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not platforms or platforms == "all":
            self.platforms = self.PLATFORMS
        else:
            self.platforms = tuple(p.strip() for p in platforms.split(",") if p.strip())
        unknown = set(self.platforms) - set(self.PLATFORMS)
        if unknown:
            raise ValueError(f"Unknown platform(s) {sorted(unknown)}; choose from {self.PLATFORMS}")
        self.max_pages = int(max_pages) if max_pages else None

    @property
    def render_js(self) -> bool:
        return bool(getattr(self, "crawler", None)) and self.crawler.settings.getbool("PLAYWRIGHT_ENABLED")

    # ------------------------------------------------------------------ #
    # Entry points
    # ------------------------------------------------------------------ #
    async def start(self):  # Scrapy >= 2.13
        for request in self.start_requests():
            yield request

    def start_requests(self):  # Scrapy < 2.13
        starters = {
            "books_toscrape": self._start_books,
            "webscraper_io": self._start_webscraper,
            "web_scraping_dev": self._start_web_scraping_dev,
        }
        for platform in self.platforms:
            yield from starters[platform]()

    def _within_page_budget(self, page: int) -> bool:
        return self.max_pages is None or page <= self.max_pages

    # ------------------------------------------------------------------ #
    # books.toscrape.com: static pagination
    # ------------------------------------------------------------------ #
    def _start_books(self):
        yield scrapy.Request(
            "https://books.toscrape.com/catalogue/page-1.html",
            callback=self.parse_books_listing,
            cb_kwargs={"page": 1},
        )

    def parse_books_listing(self, response: Response, page: int):
        for href in response.css("article.product_pod h3 a::attr(href)").getall():
            yield response.follow(href, callback=self.parse_books_product)

        next_href = response.css("li.next a::attr(href)").get()
        if next_href and self._within_page_budget(page + 1):
            yield response.follow(next_href, callback=self.parse_books_listing, cb_kwargs={"page": page + 1})

    def parse_books_product(self, response: Response):
        facts = {
            row.css("th::text").get("").strip(): row.css("td::text").get("").strip()
            for row in response.css("table.table-striped tr")
        }
        main = response.css("div.product_main")
        crumbs = response.css("ul.breadcrumb li a::text").getall()  # ["Home", "Books", "Poetry"]
        yield ProductItem(
            platform="books_toscrape",
            sku_id=facts.get("UPC"),
            title=main.css("h1::text").get(),
            url=response.url,
            category=" > ".join(crumbs),
            description=response.css("#product_description + p::text").get(),
            price=main.css("p.price_color::text").get(),
            rating=main.css("p.star-rating::attr(class)").get(),  # "star-rating Three"
            review_count=facts.get("Number of reviews"),
            in_stock=" ".join(main.css("p.availability::text").getall()),  # "In stock (22 available)"
        )

    # ------------------------------------------------------------------ #
    # webscraper.io: JSON hydration payload, or headless infinite scroll
    # ------------------------------------------------------------------ #
    def _start_webscraper(self):
        for path in WEBSCRAPER_CATEGORIES:
            category = path.replace("/", " > ")
            if self.render_js and PageMethod is not None:
                yield scrapy.Request(
                    f"{WEBSCRAPER_BASE}/scroll/{path}",
                    callback=self.parse_webscraper_rendered,
                    cb_kwargs={"category": category},
                    meta={
                        "playwright": True,
                        "download_timeout": 90,  # a full scroll of 117 cards takes ~20s
                        "playwright_page_methods": [
                            PageMethod("wait_for_selector", "div.thumbnail a.title"),
                            PageMethod("evaluate", INFINITE_SCROLL_JS),
                        ],
                    },
                )
            else:
                if self.render_js:
                    self.logger.warning("PLAYWRIGHT_ENABLED but scrapy-playwright is not installed; using JSON path")
                yield scrapy.Request(
                    f"{WEBSCRAPER_BASE}/ajax/{path}",
                    callback=self.parse_webscraper_listing,
                    cb_kwargs={"category": category},
                )

    def parse_webscraper_listing(self, response: Response, category: str):
        """Read the ``data-items`` payload the page's JavaScript would have rendered."""
        payload = response.css("[data-items]::attr(data-items)").get()
        if not payload:
            self.logger.warning("No data-items payload on %s; falling back to rendered cards", response.url)
            yield from self.parse_webscraper_rendered(response, category)
            return
        for product in json.loads(payload):
            yield response.follow(
                f"{WEBSCRAPER_BASE}/ajax/product/{product['id']}",
                callback=self.parse_webscraper_product,
                cb_kwargs={"category": category, "listing": product},
            )

    def parse_webscraper_rendered(self, response: Response, category: str):
        """Parse a DOM that was rendered by the browser (or served statically)."""
        for card in response.css("div.thumbnail"):
            href = card.css("a.title::attr(href)").get()
            if not href:
                continue
            listing = {
                "title": card.css("a.title::attr(title)").get() or card.css("a.title::text").get(),
                "price": card.css('[itemprop="price"]::text, h4.price::text').get(),
                "description": card.css("p.description::text").get(),
            }
            yield response.follow(
                href,
                callback=self.parse_webscraper_product,
                cb_kwargs={"category": category, "listing": listing},
            )

    def parse_webscraper_product(self, response: Response, category: str, listing: dict | None = None):
        listing = listing or {}
        sku = re.search(r"/product/(\d+)", response.url)
        # Unavailable variants are rendered as disabled swatch buttons.
        swatches = response.css(".swatches button.swatch")
        in_stock = not swatches or any("disabled" not in swatch.attrib for swatch in swatches)
        yield ProductItem(
            platform="webscraper_io",
            sku_id=sku.group(1) if sku else None,
            title=response.css(".caption .title::text").get() or listing.get("title"),
            url=response.url,
            category=category,
            description=response.css('.caption [itemprop="description"]::text').get() or listing.get("description"),
            price=response.css('.caption [itemprop="price"]::text').get() or listing.get("price"),
            currency=response.css('[itemprop="priceCurrency"]::attr(content)').get(),
            rating=len(response.css(".ratings .ws-icon-star")),
            review_count=response.css('[itemprop="reviewCount"]::text').get(),
            in_stock=in_stock,
        )

    # ------------------------------------------------------------------ #
    # web-scraping.dev: JSON-LD + hidden review JSON
    # ------------------------------------------------------------------ #
    def _start_web_scraping_dev(self):
        for category in WEB_SCRAPING_DEV_CATEGORIES:
            yield scrapy.Request(
                f"https://web-scraping.dev/products?category={category}",
                callback=self.parse_wsd_listing,
                cb_kwargs={"category": category, "page": 1},
                meta=WSD_META,
            )

    def parse_wsd_listing(self, response: Response, category: str, page: int):
        links = response.css("div.products div.product h3 a::attr(href)").getall()
        for href in links:
            yield response.follow(
                href, callback=self.parse_wsd_product, cb_kwargs={"category": category}, meta=WSD_META
            )

        # The ">" link always points somewhere, so stop on the first empty page.
        next_href = next(
            (a.attrib.get("href") for a in response.css("div.paging a") if a.css("::text").get("").strip() == ">"),
            None,
        )
        next_page = int(parse_qs(urlparse(next_href or "").query).get("page", ["0"])[0])
        if links and next_href and next_page > page and self._within_page_budget(page + 1):
            yield response.follow(
                next_href,
                callback=self.parse_wsd_listing,
                cb_kwargs={"category": category, "page": next_page},
                meta=WSD_META,
            )

    def parse_wsd_product(self, response: Response, category: str):
        product = find_json_ld(response, "Product")
        offers = product.get("offers") or {}
        if isinstance(offers, list):
            offers = offers[0] if offers else {}
        aggregate = product.get("aggregateRating") or {}
        features = {
            row.css(".feature-label::text").get("").strip().lower(): row.css(".feature-value::text").get("").strip()
            for row in response.css("tr.feature")
        }
        match = re.search(r"/product/(\d+)", response.url)
        sku = match.group(1) if match else stable_id(response.url)

        yield ProductItem(
            platform="web_scraping_dev",
            sku_id=sku,
            title=product.get("name") or response.css("h3.product-title::text").get(),
            url=response.url,
            category=category,
            brand=features.get("brand"),
            description=product.get("description") or response.css("p.product-description::text").get(),
            price=response.css("span.product-price::text").get() or offers.get("price") or offers.get("lowPrice"),
            original_price=response.css("span.product-price-full::text").get(),
            currency=offers.get("priceCurrency"),
            rating=aggregate.get("ratingValue"),
            review_count=aggregate.get("reviewCount"),
            in_stock=offers.get("availability"),
        )
        yield from self._wsd_reviews(response, sku, product)

    def _wsd_reviews(self, response: Response, sku: str, product_ld: dict):
        """Reviews come from the hidden JSON the page renders client-side, else from JSON-LD."""
        hidden = response.css("script#reviews-data::text").get()
        reviews = json.loads(hidden) if hidden else []
        if reviews:
            for review in reviews:
                yield ReviewItem(
                    platform="web_scraping_dev",
                    sku_id=sku,
                    review_id=review.get("id"),
                    rating=review.get("rating"),
                    review_text=review.get("text"),
                    date=review.get("date"),
                    author=review.get("author"),
                )
            return
        for review in product_ld.get("review") or []:
            body = review.get("reviewBody")
            yield ReviewItem(
                platform="web_scraping_dev",
                sku_id=sku,
                review_id=stable_id(sku, review.get("datePublished"), body),
                rating=(review.get("reviewRating") or {}).get("ratingValue"),
                review_text=body,
                date=review.get("datePublished"),
                author=review.get("author"),
            )
