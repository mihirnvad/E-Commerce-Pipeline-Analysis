"""Pydantic v2 schemas: the single source of truth for every document in MongoDB.

All cleaning of raw scraped strings lives in ``mode="before"`` validators here, so
spiders stay thin extraction layers and exactly the same rules apply to live crawls
and to the synthetic seed data.

The cleaning helpers are plain functions so they can be unit-tested in isolation.
"""
from __future__ import annotations

import html
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from dateutil import parser as date_parser
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    PlainSerializer,
    computed_field,
    field_validator,
    model_validator,
)

# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")
# UTF-8 bytes mis-decoded as cp1252 show up as a lead character (Â, Ã, â, ...) followed by
# one or more characters that cp1252 maps the 0x80-0xBF continuation bytes onto.
_CP1252_CONTINUATION = bytes(range(0x80, 0xC0)).decode("cp1252", errors="ignore")
_MOJIBAKE_RE = re.compile(f"[Â-ô][{re.escape(_CP1252_CONTINUATION)}]+")


def _repair_mojibake_chunk(match: re.Match) -> str:
    chunk = match.group(0)
    for encoding in ("cp1252", "latin-1"):
        try:
            return chunk.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
    return chunk


def fix_mojibake(text: str) -> str:
    """Repair UTF-8 that was mis-decoded as cp1252/latin-1 (``"Â£51.77"`` -> ``"£51.77"``).

    Works chunk by chunk, so correctly decoded characters in the same string are untouched.
    """
    return _MOJIBAKE_RE.sub(_repair_mojibake_chunk, text)


def clean_text(value: Any) -> str:
    """Unescape entities, repair encoding, strip tags and collapse whitespace."""
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = fix_mojibake(text)
    # NFC rather than NFKC: NFKC would rewrite "™" as "TM" and "½" as "1⁄2" in titles.
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


# --------------------------------------------------------------------------- #
# Money
# --------------------------------------------------------------------------- #
# Longest symbols first so "US$" wins over "$".
_CURRENCY_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("US$", "USD"),
    ("C$", "CAD"),
    ("A$", "AUD"),
    ("$", "USD"),
    ("£", "GBP"),
    ("€", "EUR"),
    ("¥", "JPY"),
    ("₹", "INR"),
)
_ISO_CURRENCIES = frozenset({"USD", "GBP", "EUR", "CAD", "AUD", "JPY", "INR", "CNY", "CHF", "MXN"})
_ISO_CODE_RE = re.compile(r"\b([A-Z]{3})\b")
# The optional sign keeps "-5.00" negative so the model rejects it instead of storing 5.00.
_NUMBER_RE = re.compile(r"-?\d[\d.,\s ]*")


def _normalize_number(raw: str) -> float:
    num = re.sub(r"[\s ]", "", raw).rstrip(".,")
    if "," in num and "." in num:
        if num.rfind(",") > num.rfind("."):  # 1.299,99 (EU)
            num = num.replace(".", "").replace(",", ".")
        else:  # 1,299.99 (US)
            num = num.replace(",", "")
    elif "," in num:
        head, _, tail = num.rpartition(",")
        # "12,5" / "12,50" are decimal commas; "1,299" is a thousands separator.
        num = f"{head.replace(',', '')}.{tail}" if len(tail) in (1, 2) else num.replace(",", "")
    elif num.count(".") > 1:  # 1.299.999
        num = num.replace(".", "")
    return float(num)


def parse_price(value: Any) -> tuple[float | None, str | None]:
    """Split a dirty price string into ``(amount, ISO currency)``.

    >>> parse_price("$1,299.99 USD")
    (1299.99, 'USD')
    >>> parse_price("1.299,99 €")
    (1299.99, 'EUR')
    """
    if value is None or isinstance(value, bool):
        return None, None
    if isinstance(value, (int, float)):
        return float(value), None
    text = clean_text(value)
    currency = None
    iso = _ISO_CODE_RE.search(text.upper())
    if iso and iso.group(1) in _ISO_CURRENCIES:
        currency = iso.group(1)
    else:
        for symbol, code in _CURRENCY_SYMBOLS:
            if symbol in text:
                currency = code
                break
    match = _NUMBER_RE.search(text)
    if not match:
        return None, currency
    try:
        return _normalize_number(match.group(0)), currency
    except ValueError:
        return None, currency


# --------------------------------------------------------------------------- #
# Ratings, counts, stock
# --------------------------------------------------------------------------- #
_WORD_NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_WORD_NUMBER_RE = re.compile(rf"\b({'|'.join(_WORD_NUMBERS)})\b")
_FLOAT_RE = re.compile(r"\d+(?:[.,]\d+)?")


def parse_rating(value: Any) -> float | None:
    """``"4.5 out of 5 stars"`` -> 4.5, ``"star-rating Three"`` -> 3.0."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = clean_text(value).lower()
    word = _WORD_NUMBER_RE.search(text)
    if word:
        return float(_WORD_NUMBERS[word.group(1)])
    match = _FLOAT_RE.search(text)
    return float(match.group(0).replace(",", ".")) if match else None


def parse_int(value: Any) -> int | None:
    """``"1,024 reviews"`` -> 1024."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    digits = re.search(r"\d[\d,.\s]*", clean_text(value))
    if not digits:
        return None
    return int(re.sub(r"[^\d]", "", digits.group(0)))


_OUT_OF_STOCK = ("out of stock", "outofstock", "sold out", "soldout", "unavailable", "discontinued")
_IN_STOCK = ("in stock", "instock", "available", "limitedavailability", "onlineonly", "true", "yes")


def parse_availability(value: Any) -> bool | None:
    """Map free-text or schema.org availability onto a boolean."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = clean_text(value).lower()
    if any(token in text for token in _OUT_OF_STOCK):
        return False
    if any(token in text for token in _IN_STOCK):
        return True
    return None


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
_RELATIVE_RE = re.compile(
    r"(?P<n>\d+|an?|one)\s+(?P<unit>second|minute|hour|day|week|month|year)s?\s+ago", re.I
)
_UNIT_DAYS = {"second": 1 / 86400, "minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30, "year": 365}
# Tried before dateutil, which is flexible but ~50x slower per call.
_COMMON_DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y")


def _parse_absolute_date(text: str) -> datetime:
    try:
        return datetime.fromisoformat(text)  # "2022-07-22", "2024-03-01t10:00:00+02:00"
    except ValueError:
        pass
    for fmt in _COMMON_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return date_parser.parse(text)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"unparseable date: {text!r}") from exc


def parse_date(value: Any, *, now: datetime | None = None) -> datetime:
    """Parse absolute *or* relative dates into a timezone-aware UTC datetime.

    Handles ISO strings, "July 22, 2022", "Reviewed in the US on July 22, 2022",
    "3 days ago", "a month ago", "yesterday" and epoch seconds/milliseconds.
    Raises ``ValueError`` when the value cannot be interpreted.
    """
    now = now or datetime.now(timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = value / 1000 if value > 1e11 else value
        parsed = datetime.fromtimestamp(seconds, tz=timezone.utc)
    elif isinstance(value, str) and value.strip():
        text = clean_text(value).lower()
        if " on " in text:  # "reviewed in the united states on july 22, 2022"
            text = text.rsplit(" on ", 1)[1]
        text = re.sub(r"^(reviewed|posted|published|updated)\s+", "", text)
        relative = _RELATIVE_RE.search(text)
        if text in {"today", "now", "just now"}:
            parsed = now
        elif text == "yesterday":
            parsed = now - timedelta(days=1)
        elif relative:
            count = 1 if relative["n"].lower() in {"a", "an", "one"} else int(relative["n"])
            parsed = now - timedelta(days=count * _UNIT_DAYS[relative["unit"].lower()])
        else:
            parsed = _parse_absolute_date(text)
    else:
        raise ValueError(f"unparseable date: {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Category taxonomy
# --------------------------------------------------------------------------- #
CATEGORY_TAXONOMY: dict[str, tuple[str, ...]] = {
    "Electronics": (
        "electronic", "computer", "laptop", "tablet", "phone", "touch", "headphone",
        "audio", "camera", "monitor", "television", "gaming", "smartwatch",
    ),
    "Apparel": ("apparel", "clothing", "fashion", "shoe", "sneaker", "shirt", "jacket", "dress", "hoodie", "boot"),
    "Home Goods": ("home goods", "household", "kitchen", "furniture", "bedding", "decor", "garden", "cookware", "bath"),
    "Grocery": ("consumable", "grocery", "food", "beverage", "snack", "candy", "drink"),
    "Books": ("book",),
}
_CATEGORY_PATTERNS = {
    canonical: re.compile(rf"^{re.escape(canonical.lower())}$|\b(?:{'|'.join(map(re.escape, keywords))})")
    for canonical, keywords in CATEGORY_TAXONOMY.items()
}
_BREADCRUMB_NOISE = {"home", "all", "all products", "products", "shop", "catalog", "catalogue", "index"}
_BREADCRUMB_SPLIT_RE = re.compile(r"\s*(?:>|/|\||»|›)\s*")


def _tidy_segment(segment: str) -> str:
    # Keep deliberate casing ("iPhone"), title-case lower-case slugs ("laptops").
    return segment if any(ch.isupper() for ch in segment) else segment.title()


def normalize_category(raw: Any) -> tuple[str, str | None]:
    """Map a site-specific breadcrumb onto the shared top-level taxonomy.

    >>> normalize_category("Home > Computers / Laptops")
    ('Electronics', 'Laptops')
    >>> normalize_category("Books > Poetry")
    ('Books', 'Poetry')
    """
    segments = [s for s in _BREADCRUMB_SPLIT_RE.split(clean_text(raw)) if s]
    while segments and segments[0].lower() in _BREADCRUMB_NOISE:
        segments.pop(0)
    if not segments:
        return "Other", None

    top = None
    for segment in segments:
        lowered = segment.lower()
        top = next((canonical for canonical, pattern in _CATEGORY_PATTERNS.items() if pattern.search(lowered)), None)
        if top:
            break

    leaf = _tidy_segment(segments[-1])
    subcategory = None if leaf.lower() == (top or "").lower() else leaf
    return top or "Other", subcategory


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
# Validated like a URL, stored in MongoDB as a plain string.
HttpUrlStr = Annotated[HttpUrl, PlainSerializer(lambda url: str(url), return_type=str)]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _StrictBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        coerce_numbers_to_str=True,  # numeric SKUs / review ids arrive as ints from JSON payloads
        validate_default=True,
    )


class ProductModel(_StrictBase):
    """A single product listing, keyed by ``(platform, sku_id)``."""

    sku_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=500)
    platform: str = Field(pattern=r"^[a-z0-9_]+$")
    url: HttpUrlStr
    category: str = "Other"
    subcategory: str | None = None
    brand: str | None = None
    description: str | None = None
    price: float = Field(gt=0, lt=1_000_000)
    original_price: float | None = Field(default=None, gt=0, lt=1_000_000)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    rating: float = Field(default=0.0, ge=0.0, le=5.0)
    review_count: int = Field(default=0, ge=0)
    in_stock: bool = True
    stock_quantity: int | None = Field(default=None, ge=0)
    scraped_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="before")
    @classmethod
    def _normalize_raw_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)

        # "$1,299.99 USD" -> price=1299.99, currency="USD"
        detected_currency = None
        for key in ("price", "original_price"):
            if isinstance(data.get(key), str):
                amount, currency = parse_price(data[key])
                data[key] = amount
                detected_currency = detected_currency or currency
        if not data.get("currency") and detected_currency:
            data["currency"] = detected_currency

        # "In stock (22 available)" -> in_stock=True, stock_quantity=22
        availability = data.get("in_stock")
        if isinstance(availability, str):
            quantity = re.search(r"(\d+)\s+(?:available|in stock|left)", availability, re.I)
            if quantity and data.get("stock_quantity") is None:
                data["stock_quantity"] = int(quantity.group(1))

        if data.get("category") is not None:
            top, sub = normalize_category(data["category"])
            data["category"] = top
            if not data.get("subcategory"):
                data["subcategory"] = sub
        return data

    @field_validator("title", "brand", "description", "subcategory", "sku_id", mode="before")
    @classmethod
    def _clean_strings(cls, value: Any) -> Any:
        if value is None or not isinstance(value, str):
            return value
        cleaned = clean_text(value)
        return cleaned or None

    @field_validator("description")
    @classmethod
    def _truncate_description(cls, value: str | None) -> str | None:
        return value[:2000] if value else value

    @field_validator("platform", mode="before")
    @classmethod
    def _slug_platform(cls, value: Any) -> Any:
        return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") if value else value

    @field_validator("currency", mode="before")
    @classmethod
    def _upper_currency(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("price", "original_price", mode="before")
    @classmethod
    def _coerce_money(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        amount, _ = parse_price(value)
        return amount

    @field_validator("price", "original_price")
    @classmethod
    def _round_money(cls, value: float | None) -> float | None:
        return round(value, 2) if value is not None else None

    @field_validator("rating", mode="before")
    @classmethod
    def _coerce_rating(cls, value: Any) -> Any:
        parsed = parse_rating(value)
        return 0.0 if parsed is None else parsed

    @field_validator("rating")
    @classmethod
    def _round_rating(cls, value: float) -> float:
        return round(value, 2)

    @field_validator("review_count", "stock_quantity", mode="before")
    @classmethod
    def _coerce_count(cls, value: Any) -> Any:
        return parse_int(value) if isinstance(value, str) else value

    @field_validator("in_stock", mode="before")
    @classmethod
    def _coerce_stock(cls, value: Any) -> Any:
        if value is None:
            return True
        parsed = parse_availability(value)
        if parsed is None:
            raise ValueError(f"unrecognised availability: {value!r}")
        return parsed

    @field_validator("scraped_at", mode="before")
    @classmethod
    def _coerce_scraped_at(cls, value: Any) -> Any:
        return _utcnow() if value is None else parse_date(value)

    @model_validator(mode="after")
    def _drop_bogus_original_price(self) -> "ProductModel":
        # A "was" price at or below the current price is not a discount; it is noise.
        if self.original_price is not None and self.original_price <= self.price:
            self.original_price = None
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def discount_pct(self) -> float:
        if self.original_price:
            return round((self.original_price - self.price) / self.original_price * 100, 2)
        return 0.0


class ReviewModel(_StrictBase):
    """A single customer review, keyed by ``(platform, sku_id, review_id)``."""

    review_id: str = Field(min_length=1, max_length=128)
    sku_id: str = Field(min_length=1, max_length=128)
    platform: str = Field(pattern=r"^[a-z0-9_]+$")
    author: str = "Anonymous"
    rating: int = Field(ge=1, le=5)
    review_text: str = Field(min_length=1, max_length=10_000)
    verified_purchase: bool = False
    date: datetime
    scraped_at: datetime = Field(default_factory=_utcnow)

    @field_validator("review_id", "sku_id", "review_text", mode="before")
    @classmethod
    def _clean_strings(cls, value: Any) -> Any:
        return clean_text(value) if isinstance(value, str) else value

    @field_validator("author", mode="before")
    @classmethod
    def _default_author(cls, value: Any) -> str:
        if isinstance(value, dict):  # schema.org {"@type": "Person", "name": ...}
            value = value.get("name")
        cleaned = clean_text(value)
        return cleaned or "Anonymous"

    @field_validator("platform", mode="before")
    @classmethod
    def _slug_platform(cls, value: Any) -> Any:
        return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") if value else value

    @field_validator("rating", mode="before")
    @classmethod
    def _coerce_rating(cls, value: Any) -> Any:
        parsed = parse_rating(value)
        if parsed is None:
            raise ValueError(f"unrecognised rating: {value!r}")
        return int(round(parsed))

    @field_validator("verified_purchase", mode="before")
    @classmethod
    def _coerce_verified(cls, value: Any) -> Any:
        if isinstance(value, str):
            return "verified" in value.lower() or value.strip().lower() in {"true", "yes", "1"}
        return bool(value)

    @field_validator("date", "scraped_at", mode="before")
    @classmethod
    def _coerce_dates(cls, value: Any) -> Any:
        return _utcnow() if value is None else parse_date(value)

    @field_validator("date")
    @classmethod
    def _not_in_future(cls, value: datetime) -> datetime:
        if value > _utcnow() + timedelta(days=1):
            raise ValueError("review date is in the future")
        return value
