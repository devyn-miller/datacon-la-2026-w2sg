import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from w2sg.rank_normalize import rank_normalize, within_query_ranks


def test_within_query_ranks_best_score_is_rank_one():
    scores = {"a": 14.2, "b": 11.8, "c": 9.4}
    ranks = within_query_ranks(scores)
    assert ranks["a"] == 1
    assert ranks["b"] == 2
    assert ranks["c"] == 3


def test_within_query_ranks_only_covers_scored_docs():
    ranks = within_query_ranks({"a": 1.0, "b": 2.0})
    assert set(ranks) == {"a", "b"}


def test_rrf_ranks_match_deck_worked_example():
    pool = {"q1": ["a", "b", "c"]}
    weak = {"bm25": {"q1": {"a": 14.2, "b": 11.8, "c": 9.4}}}
    cfg = {"rank_normalize": {"rrf_k": 60, "temperature": 0.5}}

    result = rank_normalize(pool, weak, cfg)
    assert result.ranks["q1"]["bm25"] == {"a": 1, "b": 2, "c": 3}
    target = result.target["q1"]
    assert target["a"] > target["b"] > target["c"]
    assert np.isclose(sum(target.values()), 1.0)


def test_target_sums_to_one_per_query():
    pool = {"q1": ["a", "b", "c", "d"]}
    weak = {
        "bm25": {"q1": {"a": 5.0, "b": 3.0, "c": 1.0, "d": 0.5}},
        "dense": {"q1": {"a": 0.9, "b": 0.1, "c": 0.8, "d": 0.2}},
    }
    cfg = {"rank_normalize": {"rrf_k": 60, "temperature": 0.5}}
    result = rank_normalize(pool, weak, cfg)
    assert np.isclose(sum(result.target["q1"].values()), 1.0)


def test_agreeing_sources_reinforce_the_target():
    pool = {"q1": ["a", "b", "c"]}
    weak = {
        "bm25": {"q1": {"a": 5.0, "b": 3.0, "c": 1.0}},
        "dense": {"q1": {"a": 0.9, "b": 0.5, "c": 0.1}},
        "small_ce": {"q1": {"a": 8.0, "b": 2.0, "c": 1.0}},
    }
    cfg = {"rank_normalize": {"rrf_k": 60, "temperature": 0.5}}
    result = rank_normalize(pool, weak, cfg)
    assert result.target["q1"]["a"] > result.target["q1"]["b"] > result.target["q1"]["c"]


def test_temperature_zero_rejected():
    pool = {"q1": ["a", "b"]}
    weak = {"bm25": {"q1": {"a": 1.0, "b": 0.5}}}
    cfg = {"rank_normalize": {"rrf_k": 60, "temperature": 0.0}}
    with pytest.raises(ValueError):
        rank_normalize(pool, weak, cfg)


def test_lower_temperature_sharpens_the_target():
    pool = {"q1": ["a", "b", "c"]}
    weak = {"bm25": {"q1": {"a": 5.0, "b": 3.0, "c": 1.0}}}
    cfg_sharp = {"rank_normalize": {"rrf_k": 60, "temperature": 0.05}}
    cfg_soft = {"rank_normalize": {"rrf_k": 60, "temperature": 5.0}}

    sharp = rank_normalize(pool, weak, cfg_sharp).target["q1"]
    soft = rank_normalize(pool, weak, cfg_soft).target["q1"]
    assert sharp["a"] > soft["a"]


def test_source_with_no_scores_for_query_is_skipped_not_zeroed():
    pool = {"q1": ["a", "b"]}
    weak = {
        "bm25": {"q1": {"a": 2.0, "b": 1.0}},
        "small_ce": {"q2": {"a": 5.0, "b": 4.0}},  # scored a different query only
    }
    cfg = {"rank_normalize": {"rrf_k": 60, "temperature": 0.5}}
    result = rank_normalize(pool, weak, cfg)
    assert "small_ce" not in result.ranks["q1"]
    assert result.ranks["q1"]["bm25"] == {"a": 1, "b": 2}
