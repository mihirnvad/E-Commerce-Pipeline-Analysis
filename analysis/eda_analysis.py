"""Exploratory data analysis straight from MongoDB into pandas.

    python analysis/eda_analysis.py                        # all data in $MONGO_DATABASE
    python analysis/eda_analysis.py --exclude-synthetic    # scraped listings only
    python analysis/eda_analysis.py --review-sample 20000  # faster sentiment pass

Outputs
    reports/figures/pricing_by_category.png          price spread (IQR, outliers) per category
    reports/figures/sentiment_vs_rating.png          VADER sentiment of review text vs. star rating
    reports/figures/discount_depth_distribution.png  histogram + KDE of promotion depth
    reports/figures/popularity_by_subcategory.png    where review volume (popularity) concentrates
    reports/figures/price_trend_by_category.png      weekly median price index from price_history
    reports/eda_summary.json                         every number quoted in the README
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: containers and CI have no display

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "scraper_engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scraper_engine.storage import PRODUCTS, REVIEWS, get_client, mongo_database_name, mongo_uri  # noqa: E402

logger = logging.getLogger("eda")

FIGURES_DIR = REPO_ROOT / "reports" / "figures"
SUMMARY_PATH = REPO_ROOT / "reports" / "eda_summary.json"
ANALYSIS_CURRENCY = "USD"

# --------------------------------------------------------------------------- #
# Visual system: validated categorical palette, fixed category -> colour mapping
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e6e5e1"
SERIES_1 = "#2a78d6"
SERIES_1_DARK = "#184f95"
CATEGORY_COLORS = {
    "Electronics": "#2a78d6",
    "Apparel": "#eb6834",
    "Home Goods": "#1baf7a",
    "Books": "#eda100",
    "Grocery": "#e87ba4",
    "Other": INK_MUTED,
}


def apply_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 26,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "font.size": 10,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def color_for(category: str) -> str:
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["Other"])


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9.5, va="bottom")


def _footnote(fig, text: str) -> None:
    fig.text(0.01, -0.02, text, color=INK_MUTED, fontsize=8, ha="left", va="top")


def _money(value: float, _pos=None) -> str:
    return f"${value:,.0f}" if value >= 10 else f"${value:,.2f}"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
PRODUCT_FIELDS = (
    "platform", "sku_id", "title", "category", "subcategory", "brand", "price", "original_price",
    "discount_pct", "currency", "rating", "review_count", "in_stock", "is_synthetic",
)
REVIEW_FIELDS = ("platform", "sku_id", "review_id", "rating", "review_text", "verified_purchase", "date",
                 "is_synthetic")


def _source_filter(exclude_synthetic: bool) -> dict:
    return {"is_synthetic": {"$ne": True}} if exclude_synthetic else {}


def load_products(db, *, exclude_synthetic: bool = False) -> pd.DataFrame:
    projection = {field: 1 for field in PRODUCT_FIELDS} | {"_id": 0}
    df = pd.DataFrame(list(db[PRODUCTS].find(_source_filter(exclude_synthetic), projection, batch_size=5_000)))
    if df.empty:
        return pd.DataFrame(columns=list(PRODUCT_FIELDS))
    # Scraped documents have no is_synthetic field at all.
    df["is_synthetic"] = df["is_synthetic"].eq(True) if "is_synthetic" in df else False
    df["subcategory"] = df["subcategory"].fillna("Unspecified")
    df["discount_pct"] = df["discount_pct"].fillna(0.0)
    return df


def load_reviews(db, *, exclude_synthetic: bool = False, sample: int | None = None, seed: int = 7) -> pd.DataFrame:
    projection = {field: 1 for field in REVIEW_FIELDS} | {"_id": 0}
    query = _source_filter(exclude_synthetic)
    if sample:
        pipeline = [{"$match": query}, {"$sample": {"size": sample}}, {"$project": projection}]
        docs = list(db[REVIEWS].aggregate(pipeline, allowDiskUse=True))
    else:
        docs = list(db[REVIEWS].find(query, projection, batch_size=10_000))
    df = pd.DataFrame(docs)
    return df if not df.empty else pd.DataFrame(columns=list(REVIEW_FIELDS))


def load_weekly_price_index(db, *, exclude_synthetic: bool = False) -> pd.DataFrame:
    """Median weekly price index per category, computed server-side.

    Each observation is divided by that product's first observed price, so the index is
    comparable across a $20 T-shirt and an $850 laptop. Requires MongoDB >= 7.0 ($median).
    """
    pipeline = [
        {"$match": {**_source_filter(exclude_synthetic), "currency": ANALYSIS_CURRENCY, "price_history.1": {
            "$exists": True}}},
        {"$project": {"category": 1, "price_history": 1, "base": {"$first": "$price_history.price"}}},
        {"$unwind": "$price_history"},
        {"$group": {
            "_id": {
                "category": "$category",
                "week": {"$dateTrunc": {"date": "$price_history.observed_at", "unit": "week"}},
            },
            "index": {"$median": {
                "input": {"$multiply": [100, {"$divide": ["$price_history.price", "$base"]}]},
                "method": "approximate",
            }},
            "observations": {"$sum": 1},
        }},
        {"$sort": {"_id.category": 1, "_id.week": 1}},
    ]
    rows = [
        {"category": r["_id"]["category"], "week": r["_id"]["week"], "index": r["index"],
         "observations": r["observations"]}
        for r in db[PRODUCTS].aggregate(pipeline, allowDiskUse=True)
    ]
    return pd.DataFrame(rows, columns=["category", "week", "index", "observations"])


# --------------------------------------------------------------------------- #
# Metrics (pure functions, unit tested)
# --------------------------------------------------------------------------- #
def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman's rho as Pearson on ranks (avoids pulling in scipy for one statistic)."""
    return float(a.rank().corr(b.rank()))


def pricing_summary(products: pd.DataFrame) -> pd.DataFrame:
    """Distribution metrics per category, in the analysis currency."""
    priced = products[products["currency"] == ANALYSIS_CURRENCY]
    grouped = priced.groupby("category")
    summary = pd.DataFrame({
        "listings": grouped["price"].size(),
        "mean": grouped["price"].mean(),
        "median": grouped["price"].median(),
        "q1": grouped["price"].quantile(0.25),
        "q3": grouped["price"].quantile(0.75),
        "p95": grouped["price"].quantile(0.95),
        "share_discounted_pct": grouped["discount_pct"].apply(lambda s: 100 * (s > 0).mean()),
        "median_discount_pct": grouped["discount_pct"].apply(lambda s: s[s > 0].median() if (s > 0).any() else 0.0),
    })
    summary["iqr"] = summary["q3"] - summary["q1"]
    return summary.sort_values("median", ascending=False).round(2)


def discount_summary(products: pd.DataFrame) -> dict:
    discounted = products.loc[products["discount_pct"] > 0, "discount_pct"]
    if discounted.empty:
        return {"share_discounted_pct": 0.0}
    nearest = discounted.round(0)
    round_numbers = nearest.isin([10, 15, 20, 25, 30, 40, 50]) & ((discounted - nearest).abs() <= 0.1)
    return {
        "share_discounted_pct": round(100 * len(discounted) / max(len(products), 1), 1),
        "median_depth_pct": round(float(discounted.median()), 1),
        "mean_depth_pct": round(float(discounted.mean()), 1),
        "p90_depth_pct": round(float(discounted.quantile(0.9)), 1),
        "share_round_number_promos_pct": round(100 * float(round_numbers.mean()), 1),
    }


def build_sentiment_analyzer():
    """NLTK's VADER, downloading its lexicon on first use."""
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def add_sentiment(reviews: pd.DataFrame, analyzer) -> pd.DataFrame:
    """Adds ``sentiment`` = VADER compound polarity in [-1, 1]."""
    reviews = reviews.copy()
    reviews["sentiment"] = reviews["review_text"].astype(str).map(
        lambda text: analyzer.polarity_scores(text)["compound"]
    )
    return reviews


def sentiment_summary(reviews: pd.DataFrame) -> dict:
    if reviews.empty or reviews["rating"].nunique() < 2:
        return {"reviews": int(len(reviews))}
    by_star = reviews.groupby("rating")["sentiment"].mean().round(3)
    low = reviews[reviews["rating"] <= 2]
    high = reviews[reviews["rating"] >= 4]
    return {
        "reviews": int(len(reviews)),
        "pearson_r": round(float(reviews["rating"].corr(reviews["sentiment"], method="pearson")), 3),
        "spearman_rho": round(spearman(reviews["rating"], reviews["sentiment"]), 3),
        "mean_sentiment_by_star": {int(k): float(v) for k, v in by_star.items()},
        "star_distribution_pct": {
            int(k): round(float(v), 1)
            for k, v in (100 * reviews["rating"].value_counts(normalize=True)).sort_index().items()
        },
        # Disagreement between words and stars: sarcasm, mis-clicks, shipping complaints.
        "low_star_but_positive_text_pct": round(100 * float((low["sentiment"] > 0.05).mean()), 1) if len(low) else None,
        "high_star_but_negative_text_pct": round(100 * float((high["sentiment"] < -0.05).mean()), 1)
        if len(high) else None,
    }


def popularity_summary(products: pd.DataFrame) -> dict:
    counts = products["review_count"].fillna(0).sort_values(ascending=False).to_numpy()
    total = counts.sum()
    if total == 0:
        return {"total_reviews": 0}

    def top_share(fraction: float) -> float:
        k = max(1, int(len(counts) * fraction))
        return round(100 * counts[:k].sum() / total, 1)

    reviewed = products[products["review_count"] > 0]
    by_sub = (products.groupby(["category", "subcategory"])["review_count"].sum()
              .sort_values(ascending=False).head(5))
    return {
        "total_reviews": int(total),
        "median_reviews_per_listing": float(np.median(counts)),
        "top_1pct_share_of_reviews_pct": top_share(0.01),
        "top_10pct_share_of_reviews_pct": top_share(0.10),
        "listings_without_reviews_pct": round(100 * float((counts == 0).mean()), 1),
        "rating_vs_log_reviews_spearman": round(spearman(reviewed["rating"], reviewed["review_count"]), 3)
        if len(reviewed) > 2 else None,
        "top_subcategories": [
            {"category": cat, "subcategory": sub, "reviews": int(n)} for (cat, sub), n in by_sub.items()
        ],
    }


def trend_summary(index: pd.DataFrame) -> dict:
    result = {}
    for category, rows in index.groupby("category"):
        rows = rows.sort_values("week")
        if len(rows) >= 2:
            result[category] = {
                "weeks": int(len(rows)),
                "change_pct": round(float(rows["index"].iloc[-1] - rows["index"].iloc[0]), 1),
            }
    return result


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_pricing_by_category(products: pd.DataFrame, path: Path, footnote: str) -> None:
    priced = products[products["currency"] == ANALYSIS_CURRENCY]
    order = priced.groupby("category")["price"].median().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    sns.boxplot(
        data=priced, x="category", y="price", order=order, hue="category", hue_order=order, legend=False,
        palette={c: color_for(c) for c in order}, width=0.5, linewidth=1.1, log_scale=True, saturation=1,
        boxprops={"alpha": 0.85, "edgecolor": INK_SECONDARY}, whiskerprops={"color": INK_SECONDARY},
        capprops={"color": INK_SECONDARY}, medianprops={"color": INK, "linewidth": 2},
        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.25, "markerfacecolor": INK_MUTED,
                    "markeredgewidth": 0},
        ax=ax,
    )
    medians = priced.groupby("category")["price"].median()
    for position, category in enumerate(order):
        ax.annotate(f"median {_money(medians[category])}", (position + 0.28, medians[category]),
                    color=INK, fontsize=9, va="center")
    counts = priced["category"].value_counts()
    ax.set_xticks(range(len(order)), [f"{c}\nn = {counts[c]:,}" for c in order])
    ax.yaxis.set_major_formatter(FuncFormatter(_money))
    ax.set(xlabel="", ylabel=f"Price ({ANALYSIS_CURRENCY}, log scale)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Price distribution by category")
    _subtitle(ax, "Box = interquartile range (middle 50% of listings), line = median, "
                  "dots = outliers beyond 1.5 × IQR")
    _footnote(fig, footnote)
    fig.savefig(path)
    plt.close(fig)


def plot_sentiment_vs_rating(reviews: pd.DataFrame, stats: dict, path: Path, footnote: str, seed: int = 7) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    shown = reviews.sample(min(len(reviews), 6_000), random_state=seed)
    rng = np.random.default_rng(seed)
    ax.scatter(shown["rating"] + rng.uniform(-0.22, 0.22, len(shown)), shown["sentiment"], s=7, alpha=0.18,
               color=SERIES_1, linewidths=0, label="Individual reviews (sample)")
    sns.regplot(data=reviews, x="rating", y="sentiment", scatter=False, ci=None, ax=ax,
                line_kws={"color": SERIES_1_DARK, "linewidth": 2, "label": "Linear fit (all reviews)"})
    means = reviews.groupby("rating")["sentiment"].mean()
    ax.scatter(means.index, means.values, s=70, color=INK, zorder=5, edgecolors=SURFACE, linewidths=2,
               label="Mean sentiment per star")
    for star, value in means.items():
        ax.annotate(f"{value:+.2f}", (star, value), xytext=(12, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    ax.set(xlabel="Star rating", ylabel="VADER compound sentiment", xticks=[1, 2, 3, 4, 5], ylim=(-1.05, 1.05),
           xlim=(0.5, 5.7))
    ax.grid(axis="x", visible=False)
    ax.set_title("Review text sentiment rises with star rating")
    _subtitle(ax, f"Pearson r = {stats['pearson_r']:.2f}, Spearman ρ = {stats['spearman_rho']:.2f} across "
                  f"{stats['reviews']:,} reviews. Points are jittered horizontally")
    handles, labels = ax.get_legend_handles_labels()
    fit = plt.Line2D([], [], color=SERIES_1_DARK, linewidth=2)
    ax.legend([*handles, fit], [*labels, "Linear fit (all reviews)"], loc="lower right", fontsize=8.5)
    _footnote(fig, footnote)
    fig.savefig(path)
    plt.close(fig)


def plot_discount_depth(products: pd.DataFrame, stats: dict, path: Path, footnote: str) -> None:
    discounted = products[products["discount_pct"] > 0]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    # Bins centred on whole 2.5% steps, so a "20% off" promo falls in the middle of one bar.
    sns.histplot(discounted["discount_pct"], binwidth=2.5, binrange=(-1.25, 81.25), color=SERIES_1, alpha=0.75,
                 edgecolor=SURFACE, linewidth=1, kde=True, line_kws={"linewidth": 2, "color": SERIES_1_DARK}, ax=ax)
    ax.lines[-1].set_color(SERIES_1_DARK)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)  # headroom so the median label clears the tallest bar
    median = stats["median_depth_pct"]
    ax.axvline(median, color=INK, linewidth=1.2)
    ax.annotate(f"median {median:.0f}% off", (median, ax.get_ylim()[1]), xytext=(6, -4),
                textcoords="offset points", fontsize=9, color=INK, va="top")
    on_sale = products.groupby("category")["discount_pct"].apply(lambda s: 100 * (s > 0).mean())
    depth = discounted.groupby("category")["discount_pct"].median()
    lines = "\n".join(
        f"{cat}: {share:.0f}% of listings on sale, median {depth[cat]:.0f}% off"
        for cat, share in on_sale.sort_values(ascending=False).items() if cat in depth
    )
    ax.text(0.98, 0.72, lines, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK_SECONDARY,
            linespacing=1.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}%"))
    ax.set(xlabel="Discount off original price", ylabel="Listings", xlim=(0, 80))
    ax.grid(axis="x", visible=False)
    ax.set_title("How deep do promotions go?")
    _subtitle(ax, f"{stats['share_discounted_pct']:.0f}% of listings are discounted. The spikes are advertised "
                  f"round-number promos (10, 15, 20 … 50% off): {stats['share_round_number_promos_pct']:.0f}% "
                  f"of all discounts")
    _footnote(fig, footnote)
    fig.savefig(path)
    plt.close(fig)


def plot_popularity(products: pd.DataFrame, stats: dict, path: Path, footnote: str, top_n: int = 12) -> None:
    by_sub = (products.groupby(["category", "subcategory"])["review_count"].sum()
              .sort_values(ascending=False).head(top_n).reset_index())
    by_sub["label"] = by_sub["subcategory"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = [color_for(c) for c in by_sub["category"]]
    ax.barh(by_sub["label"], by_sub["review_count"], color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.invert_yaxis()
    for i in range(min(3, len(by_sub))):
        value = by_sub["review_count"].iloc[i]
        ax.annotate(f"{value:,.0f}", (value, i), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / 1000:,.0f}k" if v >= 1000 else f"{v:,.0f}"))
    ax.set(xlabel="Total customer reviews (popularity proxy)", ylabel="")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_for(c)) for c in dict.fromkeys(by_sub["category"])]
    ax.legend(handles, list(dict.fromkeys(by_sub["category"])), loc="lower right", fontsize=9)
    ax.set_title(f"Where the attention goes: top {len(by_sub)} subcategories by review volume")
    _subtitle(ax, f"Popularity is extremely concentrated: the top 1% of listings hold "
                  f"{stats['top_1pct_share_of_reviews_pct']:.0f}% of all reviews, the top 10% hold "
                  f"{stats['top_10pct_share_of_reviews_pct']:.0f}%")
    _footnote(fig, footnote)
    fig.savefig(path)
    plt.close(fig)


def plot_price_trend(index: pd.DataFrame, path: Path, footnote: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axhline(100, color=INK_MUTED, linewidth=0.8)
    for category, rows in index.groupby("category"):
        rows = rows.sort_values("week")
        if len(rows) < 2:  # a single week is a point, not a trend
            continue
        ax.plot(rows["week"], rows["index"], color=color_for(category), linewidth=2, label=category)
        ax.annotate(f"{category} {rows['index'].iloc[-1]:.1f}", (rows["week"].iloc[-1], rows["index"].iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set(xlabel="", ylabel="Median price index (first week = 100)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", fontsize=9)
    ax.margins(x=0.02)
    ax.set_xlim(right=ax.get_xlim()[1] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_title("Weekly price trend by category")
    _subtitle(ax, "Each product's price relative to its first observation, median per category per week "
                  "(computed in MongoDB from price_history)")
    _footnote(fig, footnote)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def source_footnote(products: pd.DataFrame, db_name: str) -> str:
    synthetic = int(products["is_synthetic"].sum())
    scraped = len(products) - synthetic
    parts = []
    if scraped:
        parts.append(f"{scraped:,} scraped")
    if synthetic:
        parts.append(f"{synthetic:,} synthetic (scripts/seed_50k_mock.py)")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"Source: MongoDB '{db_name}', {len(products):,} listings: {' + '.join(parts) or 'none'}. Generated {today}."


def run(db, *, out_dir: Path, summary_path: Path, exclude_synthetic: bool, review_sample: int | None,
        seed: int) -> dict:
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    products = load_products(db, exclude_synthetic=exclude_synthetic)
    if products.empty:
        raise SystemExit("No products found. Seed with scripts/seed_50k_mock.py or run the spider first.")
    footnote = source_footnote(products, db.name)
    non_usd = int((products["currency"] != ANALYSIS_CURRENCY).sum())
    logger.info("Loaded %s products (%s non-%s excluded from price charts)", f"{len(products):,}", non_usd,
                ANALYSIS_CURRENCY)

    summary: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "database": db.name,
        "listings": int(len(products)),
        "synthetic_listings": int(products["is_synthetic"].sum()),
        "platforms": {k: int(v) for k, v in products["platform"].value_counts().items()},
        "excluded_non_usd_listings": non_usd,
    }

    pricing = pricing_summary(products)
    summary["pricing_by_category"] = pricing.reset_index().to_dict(orient="records")
    plot_pricing_by_category(products, out_dir / "pricing_by_category.png", footnote)

    usd = products[products["currency"] == ANALYSIS_CURRENCY]
    summary["discounts"] = discount_summary(usd)
    if summary["discounts"]["share_discounted_pct"] > 0:
        plot_discount_depth(usd, summary["discounts"], out_dir / "discount_depth_distribution.png", footnote)

    summary["popularity"] = popularity_summary(products)
    if summary["popularity"]["total_reviews"]:
        plot_popularity(products, summary["popularity"], out_dir / "popularity_by_subcategory.png", footnote)

    reviews = load_reviews(db, exclude_synthetic=exclude_synthetic, sample=review_sample, seed=seed)
    # A review syndicated across product variants is one opinion; score its text once.
    reviews = reviews.drop_duplicates(subset=["platform", "review_id"])
    if len(reviews) >= 10:
        logger.info("Scoring sentiment for %s reviews with VADER", f"{len(reviews):,}")
        reviews = add_sentiment(reviews, build_sentiment_analyzer())
        summary["sentiment"] = sentiment_summary(reviews)
        if "pearson_r" in summary["sentiment"]:
            plot_sentiment_vs_rating(reviews, summary["sentiment"], out_dir / "sentiment_vs_rating.png",
                                     f"{footnote} Reviews: {len(reviews):,}.", seed=seed)
    else:
        logger.warning("Only %d reviews; skipping sentiment analysis", len(reviews))

    index = load_weekly_price_index(db, exclude_synthetic=exclude_synthetic)
    summary["price_trend"] = trend_summary(index)
    if summary["price_trend"]:
        plot_price_trend(index, out_dir / "price_trend_by_category.png", footnote)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def print_report(summary: dict) -> None:
    print(f"\n=== EDA summary: {summary['listings']:,} listings "
          f"({summary['synthetic_listings']:,} synthetic) ===")
    pricing = pd.DataFrame(summary["pricing_by_category"]).set_index("category")
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        columns = ["listings", "mean", "median", "q1", "q3", "iqr", "share_discounted_pct", "median_discount_pct"]
        print("\nPricing by category (USD):\n", pricing[columns])
    for section in ("discounts", "popularity", "sentiment", "price_trend"):
        if section in summary:
            print(f"\n{section}: {json.dumps(summary[section], indent=2, default=str)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uri", default=mongo_uri())
    parser.add_argument("--db", default=mongo_database_name())
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--exclude-synthetic", action="store_true", help="analyse scraped listings only")
    parser.add_argument("--review-sample", type=int, default=None, help="score a random sample of N reviews")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = get_client(args.uri)
    try:
        summary = run(client[args.db], out_dir=args.out_dir, summary_path=args.summary,
                      exclude_synthetic=args.exclude_synthetic, review_sample=args.review_sample, seed=args.seed)
    finally:
        client.close()
    print_report(summary)
    print(f"\nFigures written to {args.out_dir}\nSummary written to {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
