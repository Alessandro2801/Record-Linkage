import pandas as pd

from src.blocking.generate import pair_passes_b1, pair_passes_b2


def test_pair_passes_b1_true_on_matching_rules():
    row = pd.Series(
        {
            "year_A": 2020,
            "year_B": 2020,
            "manufacturer_A": "Toyota",
            "manufacturer_B": "toyota",
        }
    )
    assert pair_passes_b1(row) is True


def test_pair_passes_b1_false_on_different_year():
    row = pd.Series(
        {
            "year_A": 2020,
            "year_B": 2019,
            "manufacturer_A": "Toyota",
            "manufacturer_B": "Toyota",
        }
    )
    assert pair_passes_b1(row) is False


def test_pair_passes_b2_true_with_all_constraints():
    row = pd.Series(
        {
            "year_A": 2022,
            "year_B": 2022,
            "manufacturer_A": "Honda",
            "manufacturer_B": "honda",
            "model_A": "Civic",
            "model_B": "civic",
            "fuel_type_A": "gas",
            "fuel_type_B": "gas",
        }
    )
    assert pair_passes_b2(row) is True


def test_pair_passes_b2_false_on_fuel_mismatch():
    row = pd.Series(
        {
            "year_A": 2022,
            "year_B": 2022,
            "manufacturer_A": "Honda",
            "manufacturer_B": "honda",
            "model_A": "Civic",
            "model_B": "civic",
            "fuel_type_A": "gas",
            "fuel_type_B": "diesel",
        }
    )
    assert pair_passes_b2(row) is False
