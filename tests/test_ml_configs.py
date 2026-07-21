"""Static contracts for exploratory VascuTrace training configurations."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _ROOT / "configs"


def _load(name: str) -> dict[str, object]:
    payload = yaml.safe_load((_CONFIG_DIR / name).read_text())
    assert isinstance(payload, dict)
    return payload


class TestExploratoryConfigDeltas:
    def test_v6_is_v5_plus_only_hard_negative_mining(self) -> None:
        v5 = _load("train_siamese_v5exp.yaml")
        v6 = _load("train_siamese_v6exp.yaml")
        hnm = {
            "hard_negative_mining": True,
            "hard_negative_fraction": 0.35,
            "hard_negative_oversample_weight": 3.0,
            "hard_negative_warmup_epochs": 1,
            "hard_negative_score_momentum": 0.5,
        }
        expected = dict(v5)
        expected.update(hnm)
        assert v6 == expected

    def test_v7b_is_v6_plus_exact_soft_supervision_delta(self) -> None:
        v6 = _load("train_siamese_v6exp.yaml")
        v7b = _load("train_siamese_v7b.yaml")
        expected = dict(v6)
        expected.update(
            {
                "train_cache_dir": "data/processed/p6_cache_big_soft/train",
                "val_cache_dir": "data/processed/p6_cache_big_soft/val",
                "loss": "soft_combo",
                "soft_target": True,
            }
        )
        assert v7b == expected

    def test_v7soft_is_labeled_superseded_and_does_not_enable_hnm(self) -> None:
        v5 = _load("train_siamese_v5exp.yaml")
        v7soft_path = _CONFIG_DIR / "train_siamese_v7soft.yaml"
        v7soft = _load(v7soft_path.name)
        expected = dict(v5)
        expected.update(
            {
                "train_cache_dir": "data/processed/p6_cache_big_soft/train",
                "val_cache_dir": "data/processed/p6_cache_big_soft/val",
                "loss": "soft_combo",
                "soft_target": True,
            }
        )
        assert v7soft == expected
        assert "hard_negative_mining" not in v7soft
        assert "superseded" in v7soft_path.read_text().lower()

    def test_every_exploratory_config_contains_exact_permanent_warning(self) -> None:
        for name in (
            "train_siamese_v5exp.yaml",
            "train_siamese_v6exp.yaml",
            "train_siamese_v7b.yaml",
            "train_siamese_v7soft.yaml",
        ):
            assert RESEARCH_PROTOTYPE_WARNING in (_CONFIG_DIR / name).read_text()
