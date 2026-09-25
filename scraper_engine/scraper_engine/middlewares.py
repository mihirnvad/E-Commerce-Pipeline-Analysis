"""Downloader middlewares for polite, resilient crawling.

``RotateUserAgentMiddleware``
    Gives every request a realistic desktop or mobile browser identity. Chromium
    identities also carry matching ``Sec-CH-UA`` client hints, because a Chrome UA
    without them (or a Firefox UA with them) is an easy fingerprinting tell.

``ResilientRetryMiddleware``
    Handles the status codes sites use to say "slow down" (429, 503) or "go away"
    (403), plus 200-status block pages. Instead of sleeping, which would stall
    Twisted's reactor, it raises the delay of the affected download slot with jittered
    exponential backoff (or the server's ``Retry-After``), so the whole domain slows
    down. It then re-queues the request at a lower priority and lets the slot delay
    ease back to baseline as successful responses come in.
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from http import HTTPStatus

from scrapy.downloadermiddlewares.retry import get_retry_request
from scrapy.exceptions import NotConfigured
from scrapy.http import HtmlResponse

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# User-agent rotation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BrowserProfile:
    user_agent: str
    mobile: bool = False
    headers: dict[str, str] = field(default_factory=dict)


def _chromium_hints(brand: str, major: int, platform: str, mobile: bool) -> dict[str, str]:
    return {
        "Sec-CH-UA": f'"Chromium";v="{major}", "{brand}";v="{major}", "Not=A?Brand";v="24"',
        "Sec-CH-UA-Mobile": "?1" if mobile else "?0",
        "Sec-CH-UA-Platform": f'"{platform}"',
    }


def build_default_profiles() -> list[BrowserProfile]:
    """A pool of current Chrome, Edge, Firefox and Safari identities on desktop and mobile."""
    profiles: list[BrowserProfile] = []
    desktop_os = {
        "Windows": "Windows NT 10.0; Win64; x64",
        "macOS": "Macintosh; Intel Mac OS X 10_15_7",
        "Linux": "X11; Linux x86_64",
    }
    for major in (138, 139, 140, 141):
        for platform, os_token in desktop_os.items():
            ua = f"Mozilla/5.0 ({os_token}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36"
            profiles.append(BrowserProfile(ua, headers=_chromium_hints("Google Chrome", major, platform, False)))
        edge_ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{major}.0.0.0 Safari/537.36 Edg/{major}.0.0.0"
        )
        profiles.append(BrowserProfile(edge_ua, headers=_chromium_hints("Microsoft Edge", major, "Windows", False)))
        android_ua = (
            "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{major}.0.0.0 Mobile Safari/537.36"
        )
        profiles.append(
            BrowserProfile(android_ua, mobile=True, headers=_chromium_hints("Google Chrome", major, "Android", True))
        )

    firefox_os = {
        "Windows NT 10.0; Win64; x64": False,
        "Macintosh; Intel Mac OS X 10.15": False,
        "X11; Linux x86_64": False,
        "Android 14; Mobile": True,
    }
    for major in (141, 142, 143):
        for os_token, mobile in firefox_os.items():
            ua = f"Mozilla/5.0 ({os_token}; rv:{major}.0) Gecko/{major}.0 Firefox/{major}.0" if mobile else (
                f"Mozilla/5.0 ({os_token}; rv:{major}.0) Gecko/20100101 Firefox/{major}.0"
            )
            profiles.append(BrowserProfile(ua, mobile=mobile))

    for version in ("18.5", "18.6", "26.0"):
        profiles.append(BrowserProfile(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            f"(KHTML, like Gecko) Version/{version} Safari/605.1.15"
        ))
        profiles.append(BrowserProfile(
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 "
            f"(KHTML, like Gecko) Version/{version} Mobile/15E148 Safari/604.1",
            mobile=True,
        ))
    return profiles


DEFAULT_PROFILES = build_default_profiles()
_CLIENT_HINT_PREFIX = b"Sec-Ch-Ua"  # scrapy.http.Headers normalises keys to Title-Case bytes


class RotateUserAgentMiddleware:
    """Assigns a random browser identity (UA + consistent client hints) to each request.

    Settings:
        USER_AGENT_LIST          optional list of UA strings that replaces the built-in pool
        USER_AGENT_MOBILE_RATIO  share of requests that use a mobile identity (default 0.2)

    Set ``request.meta["keep_user_agent"] = True`` to opt a request out of rotation.
    """

    def __init__(self, profiles: list[BrowserProfile], mobile_ratio: float = 0.2, seed: int | None = None):
        if not profiles:
            raise NotConfigured("RotateUserAgentMiddleware needs at least one user agent")
        self.desktop = [p for p in profiles if not p.mobile]
        self.mobile = [p for p in profiles if p.mobile]
        self.mobile_ratio = mobile_ratio if self.mobile and self.desktop else float(bool(self.mobile))
        self._rng = random.Random(seed)

    @classmethod
    def from_crawler(cls, crawler):
        settings = crawler.settings
        custom = settings.getlist("USER_AGENT_LIST")
        profiles = [BrowserProfile(ua, mobile="Mobile" in ua) for ua in custom] if custom else DEFAULT_PROFILES
        return cls(profiles, mobile_ratio=settings.getfloat("USER_AGENT_MOBILE_RATIO", 0.2))

    def choose(self) -> BrowserProfile:
        pool = self.mobile if self._rng.random() < self.mobile_ratio else self.desktop
        return self._rng.choice(pool)

    def process_request(self, request, spider=None):
        if request.meta.get("keep_user_agent"):
            return None
        profile = self.choose()
        # Drop hints left over from a previous identity (e.g. on a retried request copy).
        for key in [k for k in request.headers if k.startswith(_CLIENT_HINT_PREFIX)]:
            del request.headers[key]
        request.headers["User-Agent"] = profile.user_agent
        for name, value in profile.headers.items():
            request.headers[name] = value
        request.meta["ua_mobile"] = profile.mobile
        return None


# --------------------------------------------------------------------------- #
# Backoff + retry
# --------------------------------------------------------------------------- #
_BLOCK_TITLE_RE = re.compile(
    r"attention required|access denied|just a moment|robot check|are you a (?:human|robot)|captcha|request blocked",
    re.I,
)


def parse_retry_after(value: bytes | str | None, now: datetime | None = None) -> float | None:
    """``Retry-After`` is either delta-seconds or an HTTP-date."""
    if not value:
        return None
    text = value.decode("latin-1") if isinstance(value, bytes) else str(value)
    text = text.strip()
    if text.isdigit():
        return float(text)
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - (now or datetime.now(timezone.utc))).total_seconds())


class ResilientRetryMiddleware:
    """Exponential backoff with jitter for 429/403/503 responses and 200-status block pages.

    Settings (defaults in brackets):
        RESILIENT_RETRY_ENABLED           [True]
        RESILIENT_RETRY_HTTP_CODES        [429, 403, 503]
        RESILIENT_RETRY_MAX_TIMES         [5]
        RESILIENT_RETRY_BASE_DELAY        [1.0]   seconds before the first retry
        RESILIENT_RETRY_MAX_DELAY         [60.0]  cap for any single backoff
        RESILIENT_RETRY_PRIORITY_PENALTY  [10]    priority drop per attempt
        RESILIENT_RETRY_RECOVERY_FACTOR   [0.75]  slot delay multiplier per healthy response
        RESILIENT_RETRY_DETECT_BLOCK_PAGES [True] treat CAPTCHA/interstitial pages as soft blocks
    """

    def __init__(self, crawler, seed: int | None = None):
        settings = crawler.settings
        if not settings.getbool("RESILIENT_RETRY_ENABLED", True):
            raise NotConfigured
        self.crawler = crawler
        self.http_codes = {int(code) for code in settings.getlist("RESILIENT_RETRY_HTTP_CODES", [429, 403, 503])}
        self.max_retry_times = settings.getint("RESILIENT_RETRY_MAX_TIMES", 5)
        self.base_delay = settings.getfloat("RESILIENT_RETRY_BASE_DELAY", 1.0)
        self.max_delay = settings.getfloat("RESILIENT_RETRY_MAX_DELAY", 60.0)
        self.priority_penalty = settings.getint("RESILIENT_RETRY_PRIORITY_PENALTY", 10)
        self.recovery_factor = settings.getfloat("RESILIENT_RETRY_RECOVERY_FACTOR", 0.75)
        self.detect_block_pages = settings.getbool("RESILIENT_RETRY_DETECT_BLOCK_PAGES", True)
        self._baseline_delay: dict[str, float] = {}
        self._rng = random.Random(seed)

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)

    # -- public helpers (unit tested) ------------------------------------- #
    def compute_backoff(self, attempt: int, retry_after: bytes | str | None = None) -> float:
        """Equal-jitter exponential backoff: half fixed, half random, capped at ``max_delay``.

        Keeping a fixed half guarantees the wait always grows between attempts,
        while the random half spreads retries out so they don't all land together.
        """
        server_hint = parse_retry_after(retry_after)
        if server_hint is not None:
            return min(self.max_delay, server_hint)
        ceiling = min(self.max_delay, self.base_delay * (2 ** max(0, attempt - 1)))
        return ceiling / 2 + self._rng.uniform(0, ceiling / 2)

    def is_block_page(self, response) -> bool:
        if not self.detect_block_pages or response.status != 200 or not isinstance(response, HtmlResponse):
            return False
        title = response.xpath("//title/text()").get() or ""
        return bool(_BLOCK_TITLE_RE.search(title))

    # -- Scrapy hooks ------------------------------------------------------ #
    def process_response(self, request, response, spider=None):
        spider = spider or getattr(self.crawler, "spider", None)
        slot_key, slot = self._slot_for(request)
        handled_by_spider = response.status in request.meta.get("handle_httpstatus_list", ()) or request.meta.get(
            "handle_httpstatus_all", False
        )

        blocked = response.status in self.http_codes and not handled_by_spider
        soft_blocked = not blocked and self.is_block_page(response)
        if request.meta.get("dont_retry") or not (blocked or soft_blocked):
            self._recover(slot_key, slot)
            return response

        attempt = request.meta.get("retry_times", 0) + 1
        delay = self.compute_backoff(attempt, response.headers.get("Retry-After"))
        reason = "soft_block_page" if soft_blocked else f"{response.status} {self._phrase(response.status)}"
        retry = get_retry_request(
            request,
            spider=spider,
            reason=reason,
            max_retry_times=self.max_retry_times,
            priority_adjust=-self.priority_penalty,
        )
        stats = self.crawler.stats
        if retry is None:
            stats.inc_value("resilient_retry/gave_up")
            return response

        # Force a fresh identity on the retry; the UA middleware will assign one.
        retry.headers.pop("User-Agent", None)
        retry.meta["backoff_delay"] = delay
        self._penalize(slot_key, slot, delay)
        stats.inc_value("resilient_retry/backoffs")
        stats.max_value("resilient_retry/max_backoff_seconds", round(delay, 2))
        logger.info(
            "Backing off %.1fs on %s (attempt %d/%d, reason=%s)",
            delay, slot_key or request.url, attempt, self.max_retry_times, reason,
        )
        return retry

    # -- slot management --------------------------------------------------- #
    def _slot_for(self, request):
        key = request.meta.get("download_slot")
        engine = getattr(self.crawler, "engine", None)
        downloader = getattr(engine, "downloader", None)
        if key is None or downloader is None:
            return key, None
        return key, downloader.slots.get(key)

    def _penalize(self, key, slot, delay: float) -> None:
        if slot is None:
            return
        self._baseline_delay.setdefault(key, slot.delay)
        slot.delay = min(self.max_delay, max(slot.delay, delay))

    def _recover(self, key, slot) -> None:
        if slot is None or key not in self._baseline_delay:
            return
        baseline = self._baseline_delay[key]
        slot.delay = max(baseline, slot.delay * self.recovery_factor)
        if slot.delay <= baseline:
            del self._baseline_delay[key]

    @staticmethod
    def _phrase(status: int) -> str:
        try:
            return HTTPStatus(status).phrase
        except ValueError:
            return "Unknown"
