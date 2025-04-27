"""This is a dummy example to show how to import code from src/ for testing"""

from src.module_1.module_1_meteo_api import (
    get_data_meteo_api,
    get_mean_data_monthly,
    VARIABLES,
)
import pandas as pd


def test_get_data_meteo_api():

    # Call API
    df = get_data_meteo_api("Madrid")

    # Verify data type fetched is a dict
    assert isinstance(df, pd.DataFrame), "Function does not return a 'Dataframe'"

    # Verify df has neccessary columns

    for col in VARIABLES:
        assert col in df.columns, f"Column '{col}' is missing in Dataframe"


def test_get_mean_data_monthly():
    df = get_data_meteo_api("Madrid")

    df_monthly = get_mean_data_monthly(df)

    # Verify sampled reduced nº rows
    assert len(df_monthly) < len(df), "Resampled did not reduce number of rows"

    # Verify columns are still the same
    for col in VARIABLES:
        assert (
            col in df_monthly.columns
        ), f"Column '{col}' is missing in monthly Dataframe"
