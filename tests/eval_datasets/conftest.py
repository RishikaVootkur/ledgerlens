import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def ko_submissions() -> dict[str, Any]:
    return json.loads((FIXTURES / "submissions_ko.json").read_text())


@pytest.fixture
def ko_facts() -> dict[str, Any]:
    return json.loads((FIXTURES / "companyfacts_ko.json").read_text())
