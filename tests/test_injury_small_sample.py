"""Unit tests for small-sample injury risk shrinkage and High-bucket caps."""

from ffrank.features.risk import apply_injury_bucket_cap, shrink_injury_risk_score
from config import weights as W


def test_shrink_pulls_short_sample_toward_prior():
    # Daniels-like: ~44% observed over 2 seasons should land near Med, not High.
    shrunk = shrink_injury_risk_score(0.438, seasons=2)
    assert W.INJURY_RISK_LOW_MAX < shrunk <= W.INJURY_RISK_MED_MAX
    assert shrunk < 0.438
    assert shrunk > W.INJURY_RISK_PRIOR


def test_shrink_converges_with_long_history():
    observed = 0.40
    short = shrink_injury_risk_score(observed, seasons=2)
    long = shrink_injury_risk_score(observed, seasons=10)
    # Longer history stays closer to the observed rate than a short sample.
    assert abs(long - observed) < abs(short - observed)
    assert long > short


def test_high_capped_before_min_seasons():
    assert apply_injury_bucket_cap("High", seasons=2) == W.INJURY_RISK_BUCKET_CAP_BEFORE_HIGH
    assert apply_injury_bucket_cap("Med", seasons=2) == "Med"
    assert apply_injury_bucket_cap("Low", seasons=2) == "Low"
    assert apply_injury_bucket_cap("High", seasons=3) == "High"
