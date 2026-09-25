"""Scrapy settings for the e-commerce scraping pipeline.

Anything deployment-specific is read from the environment (or a ``.env`` file at the
repo root, see ``.env.example``).
"""
import os

from scraper_engine.storage import mongo_database_name, mongo_uri

BOT_NAME = "scraper_engine"
SPIDER_MODULES = ["scraper_engine.spiders"]
NEWSPIDER_MODULE = "scraper_engine.spiders"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
FEED_EXPORT_ENCODING = "utf-8"

# Interactive debugging listeners; a batch pipeline doesn't need open ports.
TELNETCONSOLE_ENABLED = False
REMOTE_CONTROL_ENABLED = False

# --------------------------------------------------------------------------- #
# Politeness
# --------------------------------------------------------------------------- #
ROBOTSTXT_OBEY = True
CONCURRENT_REQUESTS = 16
CONCURRENT_REQUESTS_PER_DOMAIN = 8
DOWNLOAD_DELAY = 0.25
DOWNLOAD_DELAY_JITTER = 0.5  # actual delay is 0.5x-1.5x DOWNLOAD_DELAY
DOWNLOAD_TIMEOUT = 30

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 60.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 4.0
AUTOTHROTTLE_DEBUG = False

# Per-domain overrides. web-scraping.dev publishes "Crawl-delay: 2" in robots.txt,
# which Scrapy does not enforce on its own; the spider also opts these requests out of
# AutoThrottle so the delay is never lowered.
DOWNLOAD_SLOTS = {
    "web-scraping.dev": {"concurrency": 1, "delay": 2.0, "jitter": 0.25},
}

# --------------------------------------------------------------------------- #
# Anti-blocking middlewares
# --------------------------------------------------------------------------- #
DOWNLOADER_MIDDLEWARES = {
    "scrapy.downloadermiddlewares.useragent.UserAgentMiddleware": None,
    "scraper_engine.middlewares.RotateUserAgentMiddleware": 400,
    "scraper_engine.middlewares.ResilientRetryMiddleware": 540,
}
USER_AGENT_MOBILE_RATIO = 0.2

# 429 / 403 / 503 get slot-level exponential backoff from ResilientRetryMiddleware.
# Everything else transient stays with Scrapy's built-in RetryMiddleware.
RETRY_ENABLED = True
RETRY_TIMES = 2
RETRY_HTTP_CODES = [500, 502, 504, 522, 524, 408]

RESILIENT_RETRY_ENABLED = True
RESILIENT_RETRY_HTTP_CODES = [429, 403, 503]
RESILIENT_RETRY_MAX_TIMES = 5
RESILIENT_RETRY_BASE_DELAY = 1.0
RESILIENT_RETRY_MAX_DELAY = 60.0
RESILIENT_RETRY_PRIORITY_PENALTY = 10
RESILIENT_RETRY_RECOVERY_FACTOR = 0.75
RESILIENT_RETRY_DETECT_BLOCK_PAGES = True

# --------------------------------------------------------------------------- #
# Pipelines + storage
# --------------------------------------------------------------------------- #
ITEM_PIPELINES = {
    "scraper_engine.pipelines.DataCleaningPipeline": 100,
    "scraper_engine.pipelines.MongoBatchUpsertPipeline": 300,
}
MONGO_URI = mongo_uri()
MONGO_DATABASE = mongo_database_name()
MONGO_BATCH_SIZE = int(os.getenv("MONGO_BATCH_SIZE", "500"))
MONGO_FLUSH_INTERVAL = float(os.getenv("MONGO_FLUSH_INTERVAL", "5"))

# Local development cache: HTTPCACHE_ENABLED=1 replays responses from .scrapy/httpcache.
HTTPCACHE_ENABLED = os.getenv("HTTPCACHE_ENABLED", "0") == "1"
HTTPCACHE_EXPIRATION_SECS = 24 * 3600
HTTPCACHE_IGNORE_HTTP_CODES = [403, 429, 500, 502, 503, 504]

# --------------------------------------------------------------------------- #
# Headless rendering (optional): ENABLE_PLAYWRIGHT=1 and `playwright install chromium`
# --------------------------------------------------------------------------- #
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
PLAYWRIGHT_ENABLED = os.getenv("ENABLE_PLAYWRIGHT", "0").strip().lower() in {"1", "true", "yes"}


def should_abort_request(request) -> bool:
    """Skip heavy assets in the headless browser; the parsers only need the DOM."""
    return request.resource_type in {"image", "media", "font"}


if PLAYWRIGHT_ENABLED:
    # Only requests with meta={"playwright": True} go through the browser.
    DOWNLOAD_HANDLERS = {
        "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    }
    PLAYWRIGHT_BROWSER_TYPE = "chromium"
    PLAYWRIGHT_LAUNCH_OPTIONS = {"headless": True}
    PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 30_000
    PLAYWRIGHT_MAX_PAGES_PER_CONTEXT = 4
    PLAYWRIGHT_ABORT_REQUEST = should_abort_request
