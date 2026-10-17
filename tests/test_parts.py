import numpy as np
import pytest

from src.parts import PartsSimilarity, load_parts
from src.parts.similarity import normalise, parse_current


@pytest.fixture(scope="module")
def model():
    return PartsSimilarity().fit(load_parts())


def test_normalise_units_and_synonyms():
    assert normalise("Slow Blow 5 X 20mm, 250 V") == "time lag 5x20mm 250v"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("very fast blow 1.6a 250vac 1500a ir", 1.6),
        ("1500a ir 2a", 2.0),  # breaking capacity is skipped
        ("indicator clip", None),
    ],
)
def test_parse_current(text, expected):
    assert parse_current(text) == expected


def test_alternatives_exclude_self_and_are_sorted(model):
    alt = model.alternatives("A1", k=5)
    assert len(alt) == 5
    assert "A1" not in set(alt["ID"])
    assert list(alt["score"]) == sorted(alt["score"], reverse=True)


def test_alternatives_match_current_rating(model):
    alt = model.alternatives("A1", k=3)  # 1.6 A, 250 V, 5x20 mm
    assert all("1.6A" in d for d in alt["DESCRIPTION"])


def test_parts_without_description_still_get_alternatives(model):
    pid = model.parts.loc[model.parts["description_source"].eq("attributes"), "ID"].iloc[0]
    assert len(model.alternatives(pid)) == 5


def test_empty_parts_never_recommended(model):
    empty = set(model.parts.loc[model.parts["description_source"].eq("empty"), "ID"])
    alt = model.all_alternatives(5)
    assert not empty & set(alt["alternative_ID"].dropna())


def test_free_text_search(model):
    res = model.search("time lag 2A 250V 5x20mm glass", k=5)
    assert len(res) == 5 and np.all(np.diff(res["score"]) <= 0)


def test_unknown_id(model):
    with pytest.raises(KeyError):
        model.alternatives("does-not-exist")
