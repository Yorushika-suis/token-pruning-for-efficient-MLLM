import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gpu_throughput_benchmark import choose_even_cohort, partition_round_robin  # noqa: E402


def test_choose_even_cohort_is_deterministic_and_spread():
    ids = [f"v{i}" for i in range(300)]
    selected = choose_even_cohort(ids, 16)
    assert selected == choose_even_cohort(ids, 16)
    assert selected[0] == "v0"
    assert selected[-1] == "v299"
    assert len(selected) == len(set(selected)) == 16


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_partition_round_robin_is_exact(workers):
    ids = [f"v{i}" for i in range(16)]
    parts = partition_round_robin(ids, workers)
    assert len(parts) == workers
    flattened = [item for part in parts for item in part]
    assert len(flattened) == len(ids)
    assert set(flattened) == set(ids)
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_invalid_cohort_and_partition_rejected():
    with pytest.raises(ValueError):
        choose_even_cohort(["a"], 0)
    with pytest.raises(ValueError):
        partition_round_robin(["a"], 2)
