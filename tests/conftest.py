"""Shared test setup."""

import pytest

from ledgerlens.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_data_dir(request, tmp_path, monkeypatch):
    """Unit tests never read the real data/ folder (its index may be mid-build).

    Integration tests opt out: they exist to exercise the real index.
    """
    if request.node.get_closest_marker("integration"):
        yield
        return
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
