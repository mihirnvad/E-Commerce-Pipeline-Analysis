FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:/app/scraper_engine \
    NLTK_DATA=/usr/local/share/nltk_data \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && python -m nltk.downloader -d "$NLTK_DATA" vader_lexicon

# Headless Chromium is ~400 MB with its system libraries, so it is opt-in:
#   docker compose build --build-arg INSTALL_PLAYWRIGHT=true
ARG INSTALL_PLAYWRIGHT=false
RUN if [ "$INSTALL_PLAYWRIGHT" = "true" ]; then python -m playwright install --with-deps chromium; fi

COPY . .

# UID 1000 matches the first user on most Linux hosts, so the ./reports bind mount stays writable.
RUN useradd --create-home --uid 1000 pipeline \
    && mkdir -p /app/reports/figures \
    && chown -R pipeline /app
USER pipeline

# Default: build indexes, seed 50k products, run the EDA. Override to crawl, e.g.
#   docker compose run --rm -w /app/scraper_engine app scrapy crawl ecommerce
CMD ["sh", "-c", "python scripts/setup_mongo.py && python scripts/seed_50k_mock.py && python analysis/eda_analysis.py"]
