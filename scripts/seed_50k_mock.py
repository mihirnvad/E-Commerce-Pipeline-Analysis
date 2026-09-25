"""Generate and bulk-load a realistic synthetic catalogue (50,000 products by default).

Lets anyone exercise the storage layer, the indexes and the analysis at full scale in
about a minute, without spending hours crawling live sites.

    python scripts/seed_50k_mock.py                       # 50k products + ~125k reviews
    python scripts/seed_50k_mock.py --products 5000 --workers 2
    python scripts/seed_50k_mock.py --dry-run             # generate + validate only

What makes it realistic:
* log-normal prices per subcategory, with charm pricing (x.99) and brand premiums
* a mix of round-number promotions (10/20/25 % off) and continuous markdowns
* power-law review counts (a few products hold most of the reviews)
* J-shaped, positively skewed star ratings, and review text whose sentiment mostly,
  but not always, agrees with the stars (mixed, mismatched and sarcastic reviews
  are included on purpose)
* 12 weeks of price history per product with category-level drift and promo weeks

Every record is emitted in *raw, scraped-looking* form ("$1,299.99", "3 weeks ago") and
passed through the same Pydantic models as live data, so the seed also exercises the
cleaning layer. Documents carry ``is_synthetic: true``; re-running replaces only
those, never scraped data.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

try:
    from scripts import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from pymongo.errors import BulkWriteError, PyMongoError  # noqa: E402

from scraper_engine.models import ProductModel, ReviewModel  # noqa: E402
from scraper_engine.storage import (  # noqa: E402
    PRODUCTS,
    REVIEWS,
    ensure_indexes,
    get_client,
    mongo_database_name,
    mongo_uri,
)

# --------------------------------------------------------------------------- #
# Catalogue definition
# --------------------------------------------------------------------------- #
# Fictional storefronts on reserved example.com domains.
PLATFORMS = {
    "demo_marketplace": ("https://marketplace.example.com", 0.45, 1.6),  # (base url, share, review volume)
    "demo_megastore": ("https://megastore.example.com", 0.35, 1.0),
    "demo_outlet": ("https://outlet.example.com", 0.20, 0.6),
}


@dataclass(frozen=True)
class Subcategory:
    name: str
    code: str
    noun: str
    median_price: float
    price_sigma: float
    popularity: float  # relative review volume
    series: tuple[str, ...]
    variants: tuple[str, ...]


@dataclass(frozen=True)
class Category:
    share: float
    weekly_drift: float  # relative list-price change per week (price-trend chart)
    discount_rate: float  # share of products currently on promotion
    discount_depth: float  # multiplier on promotion depth
    brands: tuple[str, ...]
    subcategories: tuple[Subcategory, ...]


CATALOG: dict[str, Category] = {
    "Electronics": Category(
        share=0.40, weekly_drift=-0.0035, discount_rate=0.30, discount_depth=0.8,
        brands=("Voltix", "Auralis", "Quantra", "Lumenic", "Zephyra", "Orbitel", "Kestrix", "Nimbra"),
        subcategories=(
            Subcategory("Laptops", "LAP", "Laptop", 849, 0.40, 0.8, ("ProBook", "AirLite", "Vector", "Titan"),
                        ('14" FHD, 16GB, 512GB SSD', '15.6" QHD, 32GB, 1TB SSD', '13.3" OLED, 8GB, 256GB SSD')),
            Subcategory("Smartphones", "PHN", "Smartphone", 599, 0.50, 1.6, ("Nova", "Pulse", "Edge", "Orbit"),
                        ("128GB, Dual SIM", "256GB, 5G", "512GB, 5G")),
            Subcategory("Headphones", "HPH", "Headphones", 89, 0.75, 2.0, ("SoundCore", "Aura", "Bassline", "Hush"),
                        ("Wireless Over-Ear, ANC", "True Wireless Earbuds", "Wired Studio")),
            Subcategory("Smartwatches", "WCH", "Smartwatch", 219, 0.45, 1.1, ("Fit", "Active", "Sport", "Classic"),
                        ("41mm, GPS", "45mm, GPS + LTE", "Kids Edition")),
            Subcategory("Monitors", "MON", "Monitor", 279, 0.45, 0.7, ("ViewPro", "Crisp", "Ultra", "Gamer"),
                        ('27" 1440p 165Hz', '32" 4K IPS', '24" 1080p 75Hz')),
            Subcategory("Cameras", "CAM", "Camera", 649, 0.60, 0.5, ("Snap", "Vista", "Frame", "Pixel"),
                        ("24MP Mirrorless Kit", "Action Cam 4K", "Instant Film")),
        ),
    ),
    "Apparel": Category(
        share=0.35, weekly_drift=-0.0010, discount_rate=0.45, discount_depth=1.25,
        brands=("Threadline", "Coastal Co.", "Urban Fable", "Stridewell", "Mistral", "Evergrain", "Northbay"),
        subcategories=(
            Subcategory("T-Shirts", "TSH", "T-Shirt", 22, 0.40, 1.8, ("Classic Crew", "Slim Fit", "Oversized"),
                        ("Navy", "Heather Grey", "Black", "Olive")),
            Subcategory("Jeans", "JNS", "Jeans", 54, 0.40, 1.2, ("Straight", "Skinny", "Relaxed"),
                        ("Dark Wash", "Light Wash", "Black")),
            Subcategory("Sneakers", "SNK", "Sneakers", 89, 0.45, 1.5, ("Runner", "Court", "Trail"),
                        ("White", "Black/Gum", "Grey")),
            Subcategory("Jackets", "JKT", "Jacket", 119, 0.50, 0.8, ("Puffer", "Rain Shell", "Denim", "Fleece"),
                        ("Black", "Forest Green", "Sand")),
            Subcategory("Dresses", "DRS", "Dress", 59, 0.50, 1.0, ("Wrap", "Midi", "Maxi", "Shirt"),
                        ("Floral", "Solid Black", "Navy Polka")),
            Subcategory("Activewear", "ACT", "Leggings", 39, 0.45, 1.1, ("Studio", "Performance", "Seamless"),
                        ("High-Rise", "7/8 Length", "Pocketed")),
        ),
    ),
    "Home Goods": Category(
        share=0.25, weekly_drift=0.0015, discount_rate=0.35, discount_depth=1.0,
        brands=("Hearthly", "Oak & Ember", "Nestwell", "Brightfold", "Casa Lumo", "Driftmoor"),
        subcategories=(
            Subcategory("Cookware", "CKW", "Pan Set", 69, 0.60, 1.2, ("Nonstick", "Cast Iron", "Stainless"),
                        ("10-Piece", "3-Piece", '12" Skillet')),
            Subcategory("Bedding", "BED", "Sheet Set", 79, 0.50, 1.1, ("Percale", "Sateen", "Linen"),
                        ("Queen", "King", "Twin")),
            Subcategory("Furniture", "FUR", "Chair", 349, 0.70, 0.5, ("Lounge", "Office", "Accent", "Dining"),
                        ("Walnut", "Oak", "Charcoal Fabric")),
            Subcategory("Lighting", "LGT", "Lamp", 59, 0.60, 0.8, ("Arc", "Desk", "Globe", "Smart"),
                        ("Brass", "Matte Black", "White")),
            Subcategory("Storage", "STO", "Organizer", 34, 0.50, 0.9, ("Stackable", "Under-Bed", "Modular"),
                        ("Set of 3", "Large", "Clear")),
            Subcategory("Decor", "DEC", "Vase", 29, 0.60, 0.7, ("Ceramic", "Glass", "Woven"),
                        ("Small", "Tall", "Set of 2")),
        ),
    ),
}

PROMO_DEPTHS = np.array([10, 15, 20, 25, 30, 40, 50], dtype=float)
PROMO_WEIGHTS = np.array([0.22, 0.18, 0.22, 0.15, 0.12, 0.07, 0.04])

# --------------------------------------------------------------------------- #
# Review text
# --------------------------------------------------------------------------- #
POSITIVE = (
    "Absolutely love this {noun}!", "Great quality for the price.", "Exceeded my expectations.",
    "Works perfectly, highly recommend.", "Would definitely buy again.", "Really happy with this purchase.",
    "Best {noun} I've owned so far.", "Excellent value and it looks fantastic.", "Five stars, no complaints at all.",
    "Well made and arrived quickly.",
)
MIXED = (
    "It's okay for the price.", "Does the job, nothing special.", "Decent {noun}, but it feels a bit cheap.",
    "Good overall, though shipping took a while.", "Nice design, average performance.",
    "Not bad, but I expected a little more.", "Fine for everyday use.",
)
NEGATIVE = (
    "Terrible quality, very disappointed.", "Stopped working after two weeks.", "Would not recommend this {noun}.",
    "Complete waste of money.", "Arrived damaged and support was unhelpful.", "Cheap materials, returned it.",
    "Awful experience from start to finish.", "Broke on the first use.",
)
# Sarcasm: positive words, negative meaning. Lexicon models like VADER misread these.
SARCASTIC = (
    "Great, it broke after two days.", "Fantastic, another {noun} for the landfill.",
    "Love waiting three weeks for something that doesn't work.",
)
ASPECTS = {
    "Electronics": ("Battery life is excellent.", "Battery drains way too fast."),
    "Apparel": ("Fits true to size and feels soft.", "Runs two sizes small and the fabric is scratchy."),
    "Home Goods": ("Easy to assemble and looks great.", "Assembly instructions were useless."),
}
FILLER = (
    "I bought this as a gift.", "Used it daily for about a month now.", "Ordered it on sale.",
    "This is my second one.", "Picked it up for my new apartment.", "",
)
AUTHOR_FIRST = (
    "Alex", "Sam", "Jordan", "Taylor", "Priya", "Wei", "Maria", "Omar", "Chen", "Fatima", "Lucas", "Aisha",
    "Noah", "Emma", "Diego", "Yuki", "Ravi", "Sofia", "Liam", "Zara", "Kofi", "Ines", "Mateo", "Hana",
)
MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December")


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def charm_price(value: float) -> float:
    """Retail-style price points: 7.49 / 24.99 / 849.99."""
    if value < 10:
        return max(0.99, math.floor(value) + (0.49 if value % 1 < 0.5 else 0.99))
    if value < 200:
        return math.floor(value) + 0.99
    return round(value / 10) * 10 - 0.01


def raw_money(value: float) -> str:
    return f"${value:,.2f}"


def raw_review_date(date: datetime, now: datetime, rng: np.random.Generator) -> str | datetime:
    """Emit dates the way storefronts display them: relative, long-form, or ISO."""
    age_days = (now - date).days
    style = rng.random()
    if style < 0.30 and age_days < 60:
        return f"{max(1, age_days // 7)} weeks ago" if age_days >= 14 else f"{max(1, age_days)} days ago"
    if style < 0.65:
        return f"Reviewed on {MONTH_NAMES[date.month - 1]} {date.day}, {date.year}"
    return date


def review_text(stars: int, noun: str, category: str, rng: np.random.Generator) -> str:
    noun = noun.lower()
    pick = lambda bank: bank[rng.integers(len(bank))].format(noun=noun)  # noqa: E731
    good_aspect, bad_aspect = ASPECTS[category]
    if rng.random() < 0.06:  # stars and words disagree: rating by accident, or grading shipping, etc.
        stars = int(rng.choice([1, 5]))
    if stars == 5:
        parts = [pick(POSITIVE), pick(POSITIVE) if rng.random() < 0.5 else good_aspect]
    elif stars == 4:
        parts = [pick(POSITIVE), pick(MIXED) if rng.random() < 0.4 else good_aspect]
    elif stars == 3:
        parts = [pick(MIXED), pick(POSITIVE) if rng.random() < 0.5 else pick(NEGATIVE)]
    elif stars == 2:
        parts = [pick(NEGATIVE), pick(MIXED)]
    else:
        parts = [pick(SARCASTIC)] if rng.random() < 0.15 else [pick(NEGATIVE), bad_aspect]
    parts.append(pick(FILLER))
    order = rng.permutation(len(parts))
    return " ".join(parts[i] for i in order if parts[i])


CHUNK_SIZE = 5_000  # fixed, so output depends on the seed only, never on the worker count


def generate_catalog(
    n_products: int,
    *,
    seed: int = 42,
    mean_reviews: float = 4.0,
    max_reviews_per_product: int = 8,
    history_weeks: int = 12,
    now: datetime | None = None,
    workers: int = 1,
) -> tuple[list[dict], list[dict]]:
    """Return validated, Mongo-ready ``(products, reviews)`` documents.

    Work is split into fixed-size chunks, each with its own child seed, and optionally
    spread across processes, since Pydantic validation dominates the runtime.
    """
    now = now or datetime.now(timezone.utc)
    # Catalogue-wide traits are drawn once, so every chunk sees the same brands.
    catalog_rng = np.random.default_rng(seed)
    brand_premium = {b: float(catalog_rng.lognormal(0, 0.18)) for c in CATALOG.values() for b in c.brands}
    brand_quality = {b: float(catalog_rng.normal(0, 0.18)) for c in CATALOG.values() for b in c.brands}

    starts = list(range(0, n_products, CHUNK_SIZE))
    child_seeds = np.random.SeedSequence(seed).spawn(len(starts))
    jobs = [
        (start, min(CHUNK_SIZE, n_products - start), child_seed, now, brand_premium, brand_quality,
         mean_reviews, max_reviews_per_product, history_weeks)
        for start, child_seed in zip(starts, child_seeds)
    ]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_generate_chunk, *zip(*jobs)))
    else:
        results = [_generate_chunk(*job) for job in jobs]

    products = [doc for chunk_products, _ in results for doc in chunk_products]
    reviews = [doc for _, chunk_reviews in results for doc in chunk_reviews]
    return products, reviews


def _generate_chunk(
    start: int,
    n: int,
    seed: np.random.SeedSequence,
    now: datetime,
    brand_premium: dict[str, float],
    brand_quality: dict[str, float],
    mean_reviews: float,
    max_reviews_per_product: int,
    history_weeks: int,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)

    platforms = list(PLATFORMS)
    platform_idx = rng.choice(len(platforms), size=n, p=[PLATFORMS[p][1] for p in platforms])
    categories = list(CATALOG)
    category_idx = rng.choice(len(categories), size=n, p=[CATALOG[c].share for c in categories])

    # Log-normal list prices (scaled by the brand premium below).
    z = rng.standard_normal(n)

    # Heavy-tailed popularity: most products have a handful of reviews, a few have thousands.
    base_reviews = np.floor(rng.pareto(1.15, n) * 6)
    zero_reviews = rng.random(n) < 0.12
    quality = rng.normal(4.15, 0.45, n)

    promo_draw = rng.random(n)
    round_promo = rng.random(n) < 0.55
    promo_depth = np.where(
        round_promo,
        rng.choice(PROMO_DEPTHS, size=n, p=PROMO_WEIGHTS),
        np.clip(rng.beta(2.2, 7.0, n) * 100, 3, 70),
    )
    promo_weeks = rng.integers(1, 5, n)
    in_stock = rng.random(n) < 0.92
    stock_qty = np.where(in_stock, np.ceil(rng.gamma(2.0, 40.0, n)), 0).astype(int)
    scraped_offsets = rng.uniform(0, 72, n)  # hours

    weeks = np.arange(-(history_weeks - 1), 1)
    history_noise = 1 + rng.normal(0, 0.008, (n, history_weeks))
    past_promo = rng.random((n, history_weeks)) < 0.05
    past_promo_cut = rng.uniform(0.10, 0.25, (n, history_weeks))

    products: list[dict] = []
    reviews: list[dict] = []
    for i in range(n):
        platform = platforms[platform_idx[i]]
        base_url, _, review_volume = PLATFORMS[platform]
        category_name = categories[category_idx[i]]
        category = CATALOG[category_name]
        sub = category.subcategories[rng.integers(len(category.subcategories))]
        brand = category.brands[rng.integers(len(category.brands))]
        series = sub.series[rng.integers(len(sub.series))]
        variant = sub.variants[rng.integers(len(sub.variants))]

        list_price = charm_price(clamp(
            math.exp(math.log(sub.median_price) + sub.price_sigma * z[i]) * brand_premium[brand], 1.5, 15_000
        ))
        discounted = promo_draw[i] < category.discount_rate
        if not discounted:
            price = list_price
        elif round_promo[i]:
            # "20% off" is advertised exactly, so the sale price is not re-charmed.
            price = round(list_price * (1 - promo_depth[i] / 100), 2)
        else:
            # Markdowns: category sets how deep they go, and prices land on x.99 again.
            price = charm_price(list_price * (1 - min(80.0, promo_depth[i] * category.discount_depth) / 100))
        if price >= list_price:
            discounted, price = False, list_price

        review_count = 0 if zero_reviews[i] else int(min(40_000, base_reviews[i] * sub.popularity * review_volume))
        product_quality = clamp(float(quality[i]) + brand_quality[brand], 1.5, 4.95)
        rating = 0.0 if review_count == 0 else clamp(
            product_quality + rng.normal(0, 0.9 / math.sqrt(review_count)), 1.0, 5.0
        )

        sku = f"{sub.code}-{start + i:07d}"
        scraped_at = now - timedelta(hours=float(scraped_offsets[i]))
        raw = {
            "platform": platform,
            "sku_id": sku,
            "title": f"{brand} {series} {sub.noun} ({variant})",
            "url": f"{base_url}/p/{sku.lower()}",
            "category": f"{category_name} > {sub.name}",
            "brand": brand,
            "price": raw_money(price),
            "original_price": raw_money(list_price) if discounted else None,
            "rating": f"{rating:.1f} out of 5 stars",
            "review_count": f"{review_count:,} ratings",
            "in_stock": f"In stock ({stock_qty[i]} available)" if in_stock[i] else "Currently unavailable",
            "scraped_at": scraped_at,
        }
        doc = ProductModel.model_validate(raw).model_dump()

        # Weekly price history ending at today's (validated) price.
        list_path = list_price * (1 + category.weekly_drift) ** weeks * history_noise[i]
        path = np.where(past_promo[i], list_path * (1 - past_promo_cut[i]), list_path)
        if discounted:
            path = np.where(weeks > -promo_weeks[i], doc["price"], path)
        path[-1] = doc["price"]
        doc["price_history"] = [
            {
                "price": round(float(p), 2),
                "original_price": doc["original_price"] if (discounted and w > -promo_weeks[i]) else None,
                "observed_at": scraped_at + timedelta(weeks=int(w)),
            }
            for w, p in zip(weeks, path)
        ]
        doc["first_seen_at"] = doc["price_history"][0]["observed_at"]
        doc["last_seen_at"] = scraped_at
        doc["is_synthetic"] = True
        products.append(doc)

        n_reviews = int(min(review_count, max_reviews_per_product, rng.poisson(mean_reviews)))
        for j in range(n_reviews):
            angry_outlier = rng.random() < 0.07
            stars = 1 if angry_outlier else int(clamp(round(rng.normal(product_quality + 0.3, 0.95)), 1, 5))
            posted = scraped_at - timedelta(days=float(min(3 * 365, rng.exponential(150))))
            review = ReviewModel.model_validate({
                "platform": platform,
                "sku_id": sku,
                "review_id": f"{sku}-R{j:02d}",
                "author": f"{AUTHOR_FIRST[rng.integers(len(AUTHOR_FIRST))]} {chr(65 + rng.integers(26))}.",
                "rating": f"{stars}.0 out of 5 stars",
                "review_text": review_text(stars, sub.noun, category_name, rng),
                "verified_purchase": "Verified Purchase" if rng.random() < 0.82 else "",
                "date": raw_review_date(posted, now, rng),
                "scraped_at": scraped_at,
            }).model_dump()
            review["is_synthetic"] = True
            reviews.append(review)

    return products, reviews


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
class Progress:
    def __init__(self, label: str, total: int):
        self.label, self.total, self.done, self.started = label, total, 0, time.perf_counter()

    def advance(self, count: int) -> None:
        self.done += count
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        pct = 100 * self.done / max(self.total, 1)
        print(f"\r  {self.label:<9} {self.done:>9,}/{self.total:,} ({pct:5.1f}%)  {self.done / elapsed:>9,.0f} docs/s",
              end="", flush=True)

    def finish(self) -> float:
        print()
        return time.perf_counter() - self.started


def bulk_insert(collection, docs: list[dict], *, batch_size: int, workers: int, label: str) -> int:
    """Chunked, multithreaded ``insert_many(ordered=False)``. Returns documents inserted."""
    progress = Progress(label, len(docs))
    chunks = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]
    inserted = 0

    def insert(chunk: list[dict]) -> int:
        try:
            return len(collection.insert_many(chunk, ordered=False).inserted_ids)
        except BulkWriteError as exc:
            return exc.details.get("nInserted", 0)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(insert, chunk) for chunk in chunks]):
            count = future.result()
            inserted += count
            progress.advance(count)
    progress.finish()
    return inserted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--products", type=int, default=50_000)
    parser.add_argument("--mean-reviews", type=float, default=4.0, help="Poisson mean of reviews stored per product")
    parser.add_argument("--max-reviews-per-product", type=int, default=8)
    parser.add_argument("--history-weeks", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="processes for generation and threads for inserts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uri", default=mongo_uri())
    parser.add_argument("--db", default=mongo_database_name())
    parser.add_argument("--dry-run", action="store_true", help="generate and validate, but do not write")
    args = parser.parse_args(argv)

    print(f"Generating {args.products:,} synthetic products (seed={args.seed}) ...")
    started = time.perf_counter()
    products, reviews = generate_catalog(
        args.products,
        seed=args.seed,
        mean_reviews=args.mean_reviews,
        max_reviews_per_product=args.max_reviews_per_product,
        history_weeks=args.history_weeks,
        workers=args.workers,
    )
    print(f"  validated {len(products):,} products and {len(reviews):,} reviews "
          f"in {time.perf_counter() - started:.1f}s")
    if args.dry_run:
        return 0

    client = get_client(args.uri)
    try:
        client.admin.command("ping")
        db = client[args.db]
        ensure_indexes(db)
        removed_p = db[PRODUCTS].delete_many({"is_synthetic": True}).deleted_count
        removed_r = db[REVIEWS].delete_many({"is_synthetic": True}).deleted_count
        if removed_p or removed_r:
            print(f"Replaced previous seed ({removed_p:,} products, {removed_r:,} reviews)")

        load_started = time.perf_counter()
        n_products = bulk_insert(db[PRODUCTS], products, batch_size=args.batch_size, workers=args.workers,
                                 label=PRODUCTS)
        n_reviews = bulk_insert(db[REVIEWS], reviews, batch_size=args.batch_size, workers=args.workers,
                                label=REVIEWS)
        elapsed = time.perf_counter() - load_started
    except PyMongoError as exc:
        print(f"\nMongoDB error ({args.uri}): {exc}", file=sys.stderr)
        print("Start it with: docker compose up -d mongo", file=sys.stderr)
        return 1
    finally:
        client.close()

    total = n_products + n_reviews
    print(f"Loaded {n_products:,} products + {n_reviews:,} reviews into '{args.db}' "
          f"in {elapsed:.1f}s ({total / max(elapsed, 1e-9):,.0f} docs/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
