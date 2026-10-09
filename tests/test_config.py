import pytest

from equipoise.config import DEMO_WARNING, load_config, resolve_prices


def test_default_config_values(cfg):
    assert cfg["seed"] == 42
    assert cfg["split"]["test_fraction"] == 0.2
    assert cfg["split"]["n_folds"] == 5
    assert cfg["analysis"]["mcid_letters"] == 5
    assert cfg["analysis"]["bootstrap_reps"] == 200
    assert cfg["analysis"]["reference_arm"] == "Bevacizumab"


def test_null_prices_fall_back_to_demo():
    costs = load_config("costs")
    assert all(v is None for v in costs["prices"]["price_per_injection_inr"].values())
    p = resolve_prices(costs_cfg=costs)
    assert p.is_demo and DEMO_WARNING in p.warnings
    assert (
        p.per_injection["Bevacizumab"]
        < p.per_injection["Ranibizumab"]
        < p.per_injection["Aflibercept"]
    )


def test_override_prices():
    p = resolve_prices(
        {
            "per_injection": {"Aflibercept": 3, "Bevacizumab": 1, "Ranibizumab": 2},
            "laser_session": 4,
        }
    )
    assert not p.is_demo and p.label == "OVERRIDE" and p.laser_session == 4


def test_partial_prices_rejected():
    costs = load_config("costs")
    costs["prices"]["price_per_injection_inr"]["Bevacizumab"] = 100
    with pytest.raises(ValueError, match="partially"):
        resolve_prices(costs_cfg=costs)


def test_incomplete_override_rejected():
    with pytest.raises(ValueError):
        resolve_prices({"per_injection": {"Aflibercept": 3}, "laser_session": 1})
