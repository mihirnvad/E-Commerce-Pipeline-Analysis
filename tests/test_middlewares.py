from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest
from scrapy import Request, Spider
from scrapy.core.downloader import Slot
from scrapy.exceptions import NotConfigured
from scrapy.http import HtmlResponse
from scrapy.utils.test import get_crawler

from scraper_engine.middlewares import (
    DEFAULT_PROFILES,
    BrowserProfile,
    ResilientRetryMiddleware,
    RotateUserAgentMiddleware,
    parse_retry_after,
)

URL = "https://shop.example.com/catalogue/page-1.html"
SLOT_KEY = "shop.example.com"


class DummySpider(Spider):
    name = "dummy"


@pytest.fixture
def crawler():
    crawler = get_crawler(DummySpider, {"RESILIENT_RETRY_MAX_TIMES": 3, "RESILIENT_RETRY_BASE_DELAY": 1.0})
    crawler.spider = crawler._create_spider()
    slot = Slot(concurrency=8, delay=0.25, jitter=0)
    crawler.engine = SimpleNamespace(downloader=SimpleNamespace(slots={SLOT_KEY: slot}))
    return crawler


@pytest.fixture
def retry_mw(crawler):
    return ResilientRetryMiddleware(crawler, seed=7)


def slot_of(crawler) -> Slot:
    return crawler.engine.downloader.slots[SLOT_KEY]


def request(**meta) -> Request:
    return Request(URL, meta={"download_slot": SLOT_KEY, **meta})


def response(req: Request, status: int = 200, body: bytes = b"<html><title>Books</title></html>", headers=None):
    return HtmlResponse(req.url, status=status, body=body, headers=headers or {}, request=req)


# --------------------------------------------------------------------------- #
# RotateUserAgentMiddleware
# --------------------------------------------------------------------------- #
def test_default_pool_is_large_and_mixed():
    assert len(DEFAULT_PROFILES) >= 30
    assert any(p.mobile for p in DEFAULT_PROFILES)
    assert any(not p.mobile for p in DEFAULT_PROFILES)
    tokens = ("Chrome", "Firefox", "Edg", "Version/")
    families = {token for p in DEFAULT_PROFILES for token in tokens if token in p.user_agent}
    assert families == {"Chrome", "Firefox", "Edg", "Version/"}


def test_user_agent_rotates_across_requests():
    mw = RotateUserAgentMiddleware(DEFAULT_PROFILES, seed=1)
    agents = set()
    for _ in range(50):
        req = Request(URL)
        mw.process_request(req)
        agents.add(req.headers["User-Agent"])
    assert len(agents) > 10


def test_client_hints_match_the_chosen_browser():
    mw = RotateUserAgentMiddleware(DEFAULT_PROFILES, seed=3)
    for _ in range(200):
        req = Request(URL)
        mw.process_request(req)
        ua = req.headers["User-Agent"].decode()
        hints = req.headers.get("Sec-CH-UA")
        if "Chrome/" in ua:
            major = ua.split("Chrome/")[1].split(".")[0]
            assert hints is not None and f'v="{major}"'.encode() in hints
            assert req.headers["Sec-CH-UA-Mobile"] == (b"?1" if "Mobile" in ua else b"?0")
        else:  # Firefox and Safari do not send UA client hints
            assert hints is None


def test_stale_client_hints_are_removed_when_identity_changes():
    firefox = BrowserProfile("Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0")
    mw = RotateUserAgentMiddleware([firefox])
    req = Request(URL, headers={"Sec-CH-UA": '"Chromium";v="140"', "Sec-CH-UA-Platform": '"Windows"'})
    mw.process_request(req)
    assert b"Firefox" in req.headers["User-Agent"]
    assert "Sec-CH-UA" not in req.headers
    assert "Sec-CH-UA-Platform" not in req.headers


def test_mobile_ratio_is_respected():
    mw = RotateUserAgentMiddleware(DEFAULT_PROFILES, mobile_ratio=1.0, seed=5)
    for _ in range(20):
        req = Request(URL)
        mw.process_request(req)
        assert req.meta["ua_mobile"] is True


def test_keep_user_agent_opt_out():
    mw = RotateUserAgentMiddleware(DEFAULT_PROFILES)
    req = Request(URL, headers={"User-Agent": "pinned"}, meta={"keep_user_agent": True})
    mw.process_request(req)
    assert req.headers["User-Agent"] == b"pinned"


def test_custom_user_agent_list_setting():
    crawler = get_crawler(DummySpider, {"USER_AGENT_LIST": ["UA-one", "UA-two Mobile"]})
    mw = RotateUserAgentMiddleware.from_crawler(crawler)
    assert {p.user_agent for p in mw.desktop + mw.mobile} == {"UA-one", "UA-two Mobile"}


def test_empty_pool_is_rejected():
    with pytest.raises(NotConfigured):
        RotateUserAgentMiddleware([])


# --------------------------------------------------------------------------- #
# ResilientRetryMiddleware
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", [429, 403, 503])
def test_throttling_status_is_retried_with_priority_penalty(retry_mw, crawler, status):
    req = request()
    req.headers["User-Agent"] = "the identity that just got throttled"
    retry = retry_mw.process_response(req, response(req, status=status))
    assert isinstance(retry, Request)
    assert retry.url == URL
    assert retry.dont_filter is True
    assert retry.meta["retry_times"] == 1
    assert retry.priority == req.priority - 10
    assert "User-Agent" not in retry.headers
    assert 0.5 <= retry.meta["backoff_delay"] <= 1.0
    assert crawler.stats.get_value("resilient_retry/backoffs") == 1


def test_backoff_grows_exponentially_with_jitter(retry_mw):
    for attempt in range(1, 7):
        ceiling = min(retry_mw.max_delay, 2 ** (attempt - 1))
        samples = [retry_mw.compute_backoff(attempt) for _ in range(200)]
        assert all(ceiling / 2 <= s <= ceiling for s in samples)
        assert len({round(s, 6) for s in samples}) > 1  # jittered, not constant


def test_backoff_is_capped(retry_mw):
    assert retry_mw.compute_backoff(50) <= retry_mw.max_delay


def test_retry_after_seconds_header_wins(retry_mw):
    req = request()
    retry = retry_mw.process_response(req, response(req, status=429, headers={"Retry-After": "12"}))
    assert retry.meta["backoff_delay"] == 12.0


def test_parse_retry_after_http_date():
    now = datetime(2025, 9, 1, 12, 0, tzinfo=timezone.utc)
    assert parse_retry_after(format_datetime(now + timedelta(seconds=30), usegmt=True), now=now) == 30.0
    assert parse_retry_after(b"garbage") is None
    assert parse_retry_after(None) is None


def test_backoff_slows_the_whole_slot_then_recovers(retry_mw, crawler):
    slot = slot_of(crawler)
    req = request()
    retry_mw.process_response(req, response(req, status=429, headers={"Retry-After": "8"}))
    assert slot.delay == 8.0

    for _ in range(30):  # healthy responses ease the delay back down
        ok = request()
        retry_mw.process_response(ok, response(ok))
    assert slot.delay == pytest.approx(0.25)
    assert SLOT_KEY not in retry_mw._baseline_delay


def test_gives_up_after_max_retries(retry_mw, crawler):
    req = request(retry_times=3)
    resp = response(req, status=503)
    assert retry_mw.process_response(req, resp) is resp
    assert crawler.stats.get_value("resilient_retry/gave_up") == 1


@pytest.mark.parametrize("status", [200, 404, 500])
def test_other_statuses_pass_through(retry_mw, status):
    req = request()
    resp = response(req, status=status)
    assert retry_mw.process_response(req, resp) is resp


def test_dont_retry_and_spider_handled_statuses_pass_through(retry_mw):
    for meta in ({"dont_retry": True}, {"handle_httpstatus_list": [403]}, {"handle_httpstatus_all": True}):
        req = request(**meta)
        resp = response(req, status=403)
        assert retry_mw.process_response(req, resp) is resp


def test_soft_block_page_is_retried(retry_mw):
    req = request()
    block = response(req, body=b"<html><head><title>Just a moment...</title></head></html>")
    retry = retry_mw.process_response(req, block)
    assert isinstance(retry, Request)


def test_disabled_by_setting():
    crawler = get_crawler(DummySpider, {"RESILIENT_RETRY_ENABLED": False})
    with pytest.raises(NotConfigured):
        ResilientRetryMiddleware.from_crawler(crawler)
