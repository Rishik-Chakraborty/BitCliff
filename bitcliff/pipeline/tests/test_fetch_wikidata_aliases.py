"""Unit tests for scripts/fetch_wikidata_aliases.py's pure helper functions
(no network calls). The script itself has no package `__init__.py`, so it's
loaded here by file path, matching its standalone script status (same
pattern as scripts/enumerate_reference_files.py, which is untested)."""

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "fetch_wikidata_aliases.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_wikidata_aliases", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fwa = _load_module()


def test_chunk_splits_into_batches_of_at_most_size():
    batches = list(fwa.chunk(list(range(0, 12)), 5))
    assert batches == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]]


def test_chunk_handles_exact_multiple():
    batches = list(fwa.chunk(list(range(0, 10)), 5))
    assert batches == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


def test_chunk_empty_input():
    assert list(fwa.chunk([], 5)) == []


def test_parse_entities_response_combines_label_and_aliases_sorted_unique():
    data = {
        "entities": {
            "Q82955": {
                "id": "Q82955",
                "labels": {"en": {"language": "en", "value": "politician"}},
                "aliases": {
                    "en": [
                        {"language": "en", "value": "political leader"},
                        {"language": "en", "value": "political figure"},
                        {"language": "en", "value": "politician"},  # dupe of label
                    ]
                },
            }
        }
    }
    out = fwa.parse_entities_response(data)
    assert out == {
        "Q82955": ["political figure", "political leader", "politician"]
    }


def test_parse_entities_response_skips_missing_entities():
    data = {
        "entities": {
            "Q999999999": {"missing": ""},
            "Q1": {
                "id": "Q1",
                "labels": {"en": {"language": "en", "value": "universe"}},
                "aliases": {},
            },
        }
    }
    out = fwa.parse_entities_response(data)
    assert out == {"Q1": ["universe"]}


def test_parse_entities_response_handles_no_english_label_or_aliases():
    data = {"entities": {"Q2": {"id": "Q2", "labels": {}, "aliases": {}}}}
    out = fwa.parse_entities_response(data)
    assert out == {"Q2": []}


def test_build_url_stays_within_50_id_batches_and_includes_props():
    url = fwa.entities_url(["Q1", "Q2", "Q3"])
    assert url.startswith("https://www.wikidata.org/w/api.php?action=wbgetentities")
    assert "ids=Q1%7CQ2%7CQ3" in url or "ids=Q1|Q2|Q3" in url
    assert "props=aliases" in url
    assert "languages=en" in url
    assert "format=json" in url
