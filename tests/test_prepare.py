"""Tests del pipeline de limpieza y splits, con datos sintéticos."""

import numpy as np
import pandas as pd
import pytest

from src.data.prepare import clean, make_splits

CFG_SPLIT = {
    "seed": 42,
    "test_size": 0.15,
    "val_size": 0.15,
    "ood_model_families": ["claude"],
}


def _fake_df(n_per_group: int = 60) -> pd.DataFrame:
    """Dataset sintético: humanos + tres familias de AI, textos únicos y largos."""
    rng = np.random.default_rng(0)
    rows = []
    for model, label in [("human", 0), ("gpt", 1), ("mistral", 1), ("claude", 1)]:
        for i in range(n_per_group):
            words = rng.choice(["lima", "utec", "ensayo", "dato", "modelo"], size=50)
            rows.append(
                {
                    "text": f"{model}-{i} " + " ".join(words),
                    "label": label,
                    "prompt_name": f"p{i % 3}",
                    "source": f"src_{model}",
                    "model": model,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def df():
    return clean(_fake_df(), min_words=30)


def test_clean_removes_nulls_duplicates_and_short_texts():
    raw = _fake_df(10)
    raw.loc[0, "text"] = None
    raw.loc[1, "text"] = raw.loc[2, "text"] + "  "  # duplicado exacto tras strip
    raw.loc[3, "text"] = "muy corto"
    out = clean(raw, min_words=30)
    assert len(out) == len(raw) - 3
    assert out["text_hash"].is_unique
    assert out["text"].notna().all()


def test_splits_are_disjoint(df):
    splits = make_splits(df, CFG_SPLIT)
    hashes = [set(s["text_hash"]) for s in splits.values()]
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            assert not hashes[i] & hashes[j]
    assert sum(len(s) for s in splits.values()) == len(df)


def test_ood_families_never_in_train_or_val(df):
    splits = make_splits(df, CFG_SPLIT)
    for name in ["train", "val", "test_iid"]:
        assert "claude" not in set(splits[name]["model"])
    assert set(splits["test_ood"]["model"]) == {"claude"}
    assert (splits["test_ood"]["label"] == 1).all()


def test_stratification_keeps_class_balance(df):
    splits = make_splits(df, CFG_SPLIT)
    overall = df[df["model"] != "claude"]["label"].mean()
    for name in ["train", "val", "test_iid"]:
        assert abs(splits[name]["label"].mean() - overall) < 0.05


def test_splits_are_deterministic(df):
    a = make_splits(df, CFG_SPLIT)
    b = make_splits(df, CFG_SPLIT)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_ood_family_with_human_label_raises(df):
    bad_cfg = CFG_SPLIT | {"ood_model_families": ["human"]}
    with pytest.raises(ValueError, match="solo textos AI"):
        make_splits(df, bad_cfg)
