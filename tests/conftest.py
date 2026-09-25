"""Shared fixtures.

Storage tests run against a real MongoDB, because upsert, ``$slice`` and unique-index
semantics are exactly what they verify. Point ``MONGO_TEST_URI`` at a server
(``docker compose up -d mongo`` locally; CI uses a service container). If none is
reachable, those tests are skipped and every other test still runs.
"""
import os
import uuid

import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

MONGO_TEST_URI = os.getenv("MONGO_TEST_URI", "mongodb://localhost:27017")


@pytest.fixture(scope="session")
def mongo_client():
    client = MongoClient(MONGO_TEST_URI, serverSelectionTimeoutMS=1500, tz_aware=True)
    try:
        client.admin.command("ping")
    except PyMongoError as exc:
        client.close()
        pytest.skip(f"MongoDB not reachable at {MONGO_TEST_URI}: {exc.__class__.__name__}")
    yield client
    client.close()


@pytest.fixture
def mongo_db_name(mongo_client):
    name = f"pipeline_test_{uuid.uuid4().hex[:10]}"
    yield name
    mongo_client.drop_database(name)
