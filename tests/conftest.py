from concurrent.futures import ThreadPoolExecutor
import pytest


@pytest.fixture(scope="session")
def executor():
    return ThreadPoolExecutor()