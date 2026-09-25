"""Make ``scraper_engine`` and ``analysis`` importable when a script is run directly.

``python scripts/foo.py`` only puts ``scripts/`` on ``sys.path``; the Scrapy package lives
one level down in ``scraper_engine/``. Importing this module (first) fixes that without
requiring an editable install.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

for path in (REPO_ROOT, REPO_ROOT / "scraper_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
