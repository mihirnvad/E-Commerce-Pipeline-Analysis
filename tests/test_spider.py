"""Parser tests against trimmed copies of the real sandbox markup (no network)."""
import json

import pytest
from scrapy import Request
from scrapy.http import HtmlResponse

from scraper_engine.items import ProductItem, ReviewItem
from scraper_engine.pipelines import DataCleaningPipeline
from scraper_engine.spiders.ecommerce_spider import WSD_META, EcommerceSpider, find_json_ld


def html_response(url: str, body: str) -> HtmlResponse:
    return HtmlResponse(url=url, body=body.encode("utf-8"), encoding="utf-8", request=Request(url))


def cleaned(item):
    return DataCleaningPipeline().process_item(item)


@pytest.fixture
def spider():
    return EcommerceSpider()


BOOKS_LISTING = """
<html><body>
<ol class="row">
  <li><article class="product_pod"><h3><a href="a-light-in-the-attic_1000/index.html" title="A Light in the Attic">
    A Light in the ...</a></h3></article></li>
  <li><article class="product_pod"><h3><a href="tipping-the-velvet_999/index.html">Tipping the Velvet</a></h3>
  </article></li>
</ol>
<ul class="pager"><li class="current">Page 1 of 50</li><li class="next"><a href="page-2.html">next</a></li></ul>
</body></html>
"""

BOOKS_DETAIL = """
<html><body>
<ul class="breadcrumb">
  <li><a href="../../index.html">Home</a></li>
  <li><a href="../category/books_1/index.html">Books</a></li>
  <li><a href="../category/books/poetry_23/index.html">Poetry</a></li>
  <li class="active">A Light in the Attic</li>
</ul>
<article class="product_page">
  <div class="col-sm-6 product_main">
    <h1>A Light in the Attic</h1>
    <p class="price_color">£51.77</p>
    <p class="instock availability"><i class="icon-ok"></i> In stock (22 available) </p>
    <p class="star-rating Three"><i class="icon-star"></i></p>
  </div>
  <div id="product_description" class="sub-header"><h2>Product Description</h2></div>
  <p>It's hard to imagine a world without A Light in the Attic.</p>
  <table class="table table-striped">
    <tr><th>UPC</th><td>a897fe39b1053632</td></tr>
    <tr><th>Product Type</th><td>Books</td></tr>
    <tr><th>Availability</th><td>In stock (22 available)</td></tr>
    <tr><th>Number of reviews</th><td>0</td></tr>
  </table>
</article>
</body></html>
"""


def test_books_listing_follows_products_and_next_page(spider):
    response = html_response("https://books.toscrape.com/catalogue/page-1.html", BOOKS_LISTING)
    requests = list(spider.parse_books_listing(response, page=1))
    urls = [r.url for r in requests]
    assert urls == [
        "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
        "https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html",
        "https://books.toscrape.com/catalogue/page-2.html",
    ]
    assert requests[-1].cb_kwargs == {"page": 2}


def test_books_page_budget_stops_pagination():
    spider = EcommerceSpider(max_pages=1)
    response = html_response("https://books.toscrape.com/catalogue/page-1.html", BOOKS_LISTING)
    assert all("page-2" not in r.url for r in spider.parse_books_listing(response, page=1))


def test_books_product_parses_and_validates(spider):
    url = "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"
    [item] = spider.parse_books_product(html_response(url, BOOKS_DETAIL))
    item = cleaned(item)
    assert item["sku_id"] == "a897fe39b1053632"
    assert item["price"] == 51.77 and item["currency"] == "GBP"
    assert item["rating"] == 3.0
    assert item["stock_quantity"] == 22
    assert (item["category"], item["subcategory"]) == ("Books", "Poetry")
    assert item["description"].startswith("It's hard to imagine")


WEBSCRAPER_AJAX_LISTING = """
<html><body>
<h1 class="page-header">Computers / Laptops</h1>
<div class="row ecomerce-items ecomerce-items-ajax" data-type="ajax"
     data-items='[{&quot;id&quot;:60,&quot;title&quot;:&quot;Asus VivoBook X441NA-GA190&quot;,
     &quot;description&quot;:&quot;14\\&quot;, Celeron N3450, 4GB&quot;,&quot;price&quot;:295.99},
     {&quot;id&quot;:61,&quot;title&quot;:&quot;Prestigio SmartBook 133S&quot;,
     &quot;description&quot;:&quot;13.3\\&quot; FHD IPS&quot;,&quot;price&quot;:299}]'></div>
</body></html>
"""

WEBSCRAPER_DETAIL = """
<html><body><div class="col-lg-9">
<div class="card thumbnail" itemscope itemtype="https://schema.org/Product"><div class="caption">
  <h4 class="price float-end pull-right" itemprop="offers" itemscope itemtype="https://schema.org/Offer">
    <span itemprop="price">$1033.99</span><meta itemprop="priceCurrency" content="USD"></h4>
  <h4 class="title card-title" itemprop="name">ThinkPad Yoga</h4>
  <p class="description card-text" itemprop="description">12.5&quot; Touch, Core i3-4010U, 4GB</p>
</div>
<div class="swatches">
  <button type="button" class="btn swatch btn-primary active" value="128">128</button>
  <button type="button" class="btn swatch" value="1024" disabled>1024</button>
</div>
<div class="ratings" itemprop="aggregateRating" itemscope itemtype="https://schema.org/AggregateRating">
  <p class="review-count"><span itemprop="reviewCount">13</span> reviews
    <span class="ws-icon ws-icon-star"></span><span class="ws-icon ws-icon-star"></span></p>
</div></div></div></body></html>
"""


def test_webscraper_listing_reads_hydration_payload(spider):
    url = "https://webscraper.io/test-sites/e-commerce/ajax/computers/laptops"
    requests = list(spider.parse_webscraper_listing(html_response(url, WEBSCRAPER_AJAX_LISTING), "computers > laptops"))
    assert [r.url for r in requests] == [
        "https://webscraper.io/test-sites/e-commerce/ajax/product/60",
        "https://webscraper.io/test-sites/e-commerce/ajax/product/61",
    ]
    assert requests[0].cb_kwargs["listing"]["price"] == 295.99


def test_webscraper_listing_falls_back_to_rendered_cards(spider):
    body = """<div class="thumbnail"><a class="title" href="/test-sites/e-commerce/scroll/product/37"
              title="ThinkPad Yoga">ThinkPad Yoga</a><h4 class="price">$1033.99</h4></div>"""
    url = "https://webscraper.io/test-sites/e-commerce/scroll/computers/laptops"
    [request] = spider.parse_webscraper_listing(html_response(url, body), "computers > laptops")
    assert request.url.endswith("/scroll/product/37")
    assert request.cb_kwargs["listing"]["title"] == "ThinkPad Yoga"


def test_webscraper_product_detects_stock_from_swatches(spider):
    url = "https://webscraper.io/test-sites/e-commerce/ajax/product/37"
    [item] = spider.parse_webscraper_product(html_response(url, WEBSCRAPER_DETAIL), "computers > laptops")
    item = cleaned(item)
    assert item["sku_id"] == "37"
    assert item["title"] == "ThinkPad Yoga"
    assert (item["price"], item["currency"]) == (1033.99, "USD")
    assert item["rating"] == 2.0 and item["review_count"] == 13
    assert item["in_stock"] is True  # one enabled swatch is enough
    assert (item["category"], item["subcategory"]) == ("Electronics", "Laptops")

    sold_out = WEBSCRAPER_DETAIL.replace('value="128">', 'value="128" disabled>')
    [item] = spider.parse_webscraper_product(html_response(url, sold_out), "computers > laptops")
    assert item["in_stock"] is False


WSD_PRODUCT_LD = {
    "@context": "https://schema.org/",
    "@type": "Product",
    "name": "Box of Chocolate Candy",
    "description": "Indulge your sweet tooth with our Box of Chocolate Candy.",
    "offers": {"@type": "AggregateOffer", "priceCurrency": "USD", "lowPrice": "9.99", "highPrice": "19.99",
               "availability": "https://schema.org/InStock"},
    "aggregateRating": {"@type": "AggregateRating", "ratingValue": "4.7", "reviewCount": "10"},
    "review": [{"@type": "Review", "author": {"@type": "Person", "name": ""}, "datePublished": "2022-07-22",
                "reviewBody": "Absolutely delicious!", "reviewRating": {"ratingValue": "5"}}],
}


def wsd_page(hidden_reviews: list | None) -> str:
    hidden = (f'<script type="application/json" id="reviews-data">{json.dumps(hidden_reviews)}</script>'
              if hidden_reviews is not None else "")
    return f"""
    <html><head><script type="application/ld+json">{json.dumps(WSD_PRODUCT_LD)}</script></head><body>
    <h3 class="card-title product-title mb-3">Box of Chocolate Candy</h3>
    <div class="price"><span class="product-price mt-5 fs-1 text-success">$9.99 </span>
      <small>from <span class="product-price-full">$12.99</span></small></div>
    <table><tr class="feature"><td class="feature-label">brand</td><td class="feature-value">ChocoDelight</td></tr>
    </table>
    <div id="reviews" data-page="1"></div>
    {hidden}
    </body></html>"""


def test_web_scraping_dev_product_and_hidden_reviews(spider):
    reviews = [
        {"date": "2022-07-22", "id": "chocolate-candy-box-1", "rating": 5, "text": "Absolutely delicious!"},
        {"date": "2022-08-16", "id": "chocolate-candy-box-2", "rating": 4, "text": "Well received gift."},
    ]
    response = html_response("https://web-scraping.dev/product/1", wsd_page(reviews))
    product, *review_items = (cleaned(i) for i in spider.parse_wsd_product(response, "consumables"))

    assert isinstance(product, ProductItem)
    assert product["sku_id"] == "1"
    assert (product["price"], product["original_price"]) == (9.99, 12.99)
    assert product["discount_pct"] == pytest.approx(23.09, abs=0.01)
    assert product["brand"] == "ChocoDelight"
    assert product["rating"] == 4.7 and product["review_count"] == 10
    assert product["category"] == "Grocery"

    assert [r["review_id"] for r in review_items] == ["chocolate-candy-box-1", "chocolate-candy-box-2"]
    assert all(isinstance(r, ReviewItem) and r["author"] == "Anonymous" for r in review_items)


def test_web_scraping_dev_falls_back_to_json_ld_reviews(spider):
    response = html_response("https://web-scraping.dev/product/1", wsd_page(None))
    _product, review = spider.parse_wsd_product(response, "consumables")
    review = cleaned(review)
    assert review["review_text"] == "Absolutely delicious!"
    assert len(review["review_id"]) == 16  # deterministic content hash


def test_web_scraping_dev_listing_stops_on_empty_page(spider):
    body = """<div class="products"></div><div class="paging">
              <a href="https://web-scraping.dev/products?category=household&page=2">&gt;</a></div>"""
    response = html_response("https://web-scraping.dev/products?category=household", body)
    assert list(spider.parse_wsd_listing(response, "household", page=1)) == []


def test_web_scraping_dev_listing_paginates_with_crawl_delay_meta(spider):
    body = """<div class="products"><div class="row product"><h3><a href="https://web-scraping.dev/product/7">
              Tee</a></h3></div></div><div class="paging">
              <a href="https://web-scraping.dev/products?category=apparel&page=1">&lt;</a>
              <a href="https://web-scraping.dev/products?category=apparel&page=2">&gt;</a></div>"""
    response = html_response("https://web-scraping.dev/products?category=apparel", body)
    product_request, next_page = spider.parse_wsd_listing(response, "apparel", page=1)
    assert product_request.url == "https://web-scraping.dev/product/7"
    assert next_page.cb_kwargs == {"category": "apparel", "page": 2}
    assert next_page.meta["autothrottle_dont_adjust_delay"] is WSD_META["autothrottle_dont_adjust_delay"]


def test_find_json_ld_handles_graph_containers():
    body = """<script type="application/ld+json">{"@graph": [{"@type": "Organization"},
              {"@type": ["Product", "Thing"], "name": "Widget"}]}</script>
              <script type="application/ld+json">not json</script>"""
    assert find_json_ld(html_response("https://example.com/", body), "Product")["name"] == "Widget"


def test_unknown_platform_is_rejected():
    with pytest.raises(ValueError, match="Unknown platform"):
        EcommerceSpider(platforms="amazon")
