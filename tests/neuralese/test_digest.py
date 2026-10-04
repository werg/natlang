import json
from pathlib import Path

from natlang_neuralese.digest import PREFIX, digest_note, digest_site

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "digest-site.json").read_text())


def test_write_site_matches_the_runtime():
    site = FIXTURE["site"]
    system = FIXTURE["messages"][0]["content"]
    assert digest_site(system, site["name"], site["type"], site["value"], site["instructions"]) == FIXTURE["messages"]
    assert PREFIX == FIXTURE["prefix"] and digest_note("state") == FIXTURE["note"]
