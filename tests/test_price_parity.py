import numpy as np
import pandas as pd

from src.features.price_parity import add_parity_columns, theoretical_adr_usd


def test_ads_ratio_and_log_spread():
    assert theoretical_adr_usd(200_000, 1_000) == 20
    frame = add_parity_columns(pd.DataFrame({"adr": [22.0], "kr": [200_000.0], "fx": [1_000.0]}), adr_column="adr", kr_column="kr", fx_column="fx")
    assert np.isclose(frame.loc[0, "parity_spread"], np.log(22 / 20))
