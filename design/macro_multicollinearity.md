# Macro-feature multicollinearity analysis

Generated: 2026-11-12 14:09:35 SGT

The feature-level table estimates variance inflation factors (VIFs) from the
complete-case macro block separately at each event anchor. The group-level
table first reduces each conceptual group to its standardized first principal
component, then calculates VIF across those group scores. The diagnostic
threshold is 10; it is a screening rule, not a mechanical proof that
a variable must be removed.

## Conclusion

The analysis identifies 7 feature-level and 2 group-level threshold breaches; review the iterative reduction before fitting the macro block.

## Anchor diagnostics

| anchor_type | n_complete | n_features | condition_number | max_feature_vif | max_group_vif | retained_features |
|---|---|---|---|---|---|---|
| regular_0930 | 43 | 14 | 29.485 | 74.120 | 17.149 | spy_pre_event_return_5m, kospi_return_latest, usdkrw_return_latest, vix_level_lag1d, vix_change_lag1d, us_2y_yield_change_lag1d, us_10y_yield_change_lag1d, kosdaq150_5m_return_latest, krx300_5m_return_latest, korea_semiconductor_5m_return_latest |

## Unavailable anchors

| anchor_type | reason |
|---|---|
| premarket_dynamic | premarket_dynamic has 0 complete rows; more than 15 are required |

## Feature-level VIF

| anchor_type | feature | group | vif | exceeds_threshold |
|---|---|---|---|---|
| regular_0930 | spy_pre_event_return_5m | equity_market | 1.523 | False |
| regular_0930 | kospi_return_latest | equity_market | 2.451 | False |
| regular_0930 | usdkrw_return_latest | fx | 1.511 | False |
| regular_0930 | vix_level_lag1d | volatility | 1.485 | False |
| regular_0930 | vix_change_lag1d | volatility | 2.126 | False |
| regular_0930 | us_2y_yield_change_lag1d | rates | 3.372 | False |
| regular_0930 | us_10y_yield_change_lag1d | rates | 3.257 | False |
| regular_0930 | kospi_5m_return_latest | korea_equity_market | 64.787 | True |
| regular_0930 | kospi200_5m_return_latest | korea_equity_market | 74.120 | True |
| regular_0930 | kosdaq150_5m_return_latest | korea_equity_market | 41.877 | True |
| regular_0930 | krx300_5m_return_latest | korea_equity_market | 37.743 | True |
| regular_0930 | kosdaq_5m_return_latest | korea_equity_market | 50.727 | True |
| regular_0930 | korea_semiconductor_5m_return_latest | korea_semiconductor | 33.406 | True |
| regular_0930 | korea_semiconductor_top10_5m_return_latest | korea_semiconductor | 47.221 | True |

## Group-level VIF

| anchor_type | group | vif | exceeds_threshold |
|---|---|---|---|
| regular_0930 | equity_market | 1.565 | False |
| regular_0930 | fx | 1.349 | False |
| regular_0930 | volatility | 1.431 | False |
| regular_0930 | rates | 1.127 | False |
| regular_0930 | korea_equity_market | 16.014 | True |
| regular_0930 | korea_semiconductor | 17.149 | True |

## Feature correlations

The following Pearson correlation matrices use the same complete-case samples
as the VIF calculations.

| anchor_type | feature | spy_pre_event_return_5m | kospi_return_latest | usdkrw_return_latest | vix_level_lag1d | vix_change_lag1d | us_2y_yield_change_lag1d | us_10y_yield_change_lag1d | kospi_5m_return_latest | kospi200_5m_return_latest | kosdaq150_5m_return_latest | krx300_5m_return_latest | kosdaq_5m_return_latest | korea_semiconductor_5m_return_latest | korea_semiconductor_top10_5m_return_latest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regular_0930 | spy_pre_event_return_5m | 1.000 | -0.133 | 0.366 | 0.142 | 0.382 | -0.217 | -0.221 | 0.146 | 0.124 | 0.065 | 0.143 | 0.037 | -0.013 | 0.039 |
| regular_0930 | kospi_return_latest | -0.133 | 1.000 | -0.066 | -0.058 | -0.278 | 0.063 | 0.084 | 0.355 | 0.371 | 0.506 | 0.359 | 0.546 | 0.544 | 0.495 |
| regular_0930 | usdkrw_return_latest | 0.366 | -0.066 | 1.000 | 0.322 | 0.433 | 0.036 | 0.007 | 0.124 | 0.115 | 0.039 | 0.116 | 0.040 | 0.017 | 0.079 |
| regular_0930 | vix_level_lag1d | 0.142 | -0.058 | 0.322 | 1.000 | 0.436 | -0.155 | -0.148 | 0.054 | 0.025 | -0.066 | 0.034 | -0.036 | -0.079 | -0.028 |
| regular_0930 | vix_change_lag1d | 0.382 | -0.278 | 0.433 | 0.436 | 1.000 | -0.319 | -0.181 | 0.087 | 0.041 | -0.004 | 0.058 | -0.006 | -0.039 | 0.016 |
| regular_0930 | us_2y_yield_change_lag1d | -0.217 | 0.063 | 0.036 | -0.155 | -0.319 | 1.000 | 0.786 | -0.029 | -0.012 | 0.040 | 0.009 | 0.022 | 0.045 | 0.011 |
| regular_0930 | us_10y_yield_change_lag1d | -0.221 | 0.084 | 0.007 | -0.148 | -0.181 | 0.786 | 1.000 | 0.102 | 0.113 | 0.174 | 0.114 | 0.180 | 0.176 | 0.114 |
| regular_0930 | kospi_5m_return_latest | 0.146 | 0.355 | 0.124 | 0.054 | 0.087 | -0.029 | 0.102 | 1.000 | 0.987 | 0.904 | 0.976 | 0.900 | 0.908 | 0.945 |
| regular_0930 | kospi200_5m_return_latest | 0.124 | 0.371 | 0.115 | 0.025 | 0.041 | -0.012 | 0.113 | 0.987 | 1.000 | 0.887 | 0.981 | 0.884 | 0.917 | 0.956 |
| regular_0930 | kosdaq150_5m_return_latest | 0.065 | 0.506 | 0.039 | -0.066 | -0.004 | 0.040 | 0.174 | 0.904 | 0.887 | 1.000 | 0.879 | 0.982 | 0.930 | 0.909 |
| regular_0930 | krx300_5m_return_latest | 0.143 | 0.359 | 0.116 | 0.034 | 0.058 | 0.009 | 0.114 | 0.976 | 0.981 | 0.879 | 1.000 | 0.861 | 0.902 | 0.947 |
| regular_0930 | kosdaq_5m_return_latest | 0.037 | 0.546 | 0.040 | -0.036 | -0.006 | 0.022 | 0.180 | 0.900 | 0.884 | 0.982 | 0.861 | 1.000 | 0.933 | 0.902 |
| regular_0930 | korea_semiconductor_5m_return_latest | -0.013 | 0.544 | 0.017 | -0.079 | -0.039 | 0.045 | 0.176 | 0.908 | 0.917 | 0.930 | 0.902 | 0.933 | 1.000 | 0.971 |
| regular_0930 | korea_semiconductor_top10_5m_return_latest | 0.039 | 0.495 | 0.079 | -0.028 | 0.016 | 0.011 | 0.114 | 0.945 | 0.956 | 0.909 | 0.947 | 0.902 | 0.971 | 1.000 |

## Iterative reduction

The reduction drops the highest-VIF feature above the threshold while retaining
at least one feature from every conceptual group.

| anchor_type | step | dropped_feature | group | vif_at_drop | retained_features_after_selection |
|---|---|---|---|---|---|
| regular_0930 | 1 | kospi200_5m_return_latest | korea_equity_market | 74.120 | spy_pre_event_return_5m, kospi_return_latest, usdkrw_return_latest, vix_level_lag1d, vix_change_lag1d, us_2y_yield_change_lag1d, us_10y_yield_change_lag1d, kosdaq150_5m_return_latest, krx300_5m_return_latest, korea_semiconductor_5m_return_latest |
| regular_0930 | 2 | kosdaq_5m_return_latest | korea_equity_market | 48.298 | spy_pre_event_return_5m, kospi_return_latest, usdkrw_return_latest, vix_level_lag1d, vix_change_lag1d, us_2y_yield_change_lag1d, us_10y_yield_change_lag1d, kosdaq150_5m_return_latest, krx300_5m_return_latest, korea_semiconductor_5m_return_latest |
| regular_0930 | 3 | korea_semiconductor_top10_5m_return_latest | korea_semiconductor | 41.249 | spy_pre_event_return_5m, kospi_return_latest, usdkrw_return_latest, vix_level_lag1d, vix_change_lag1d, us_2y_yield_change_lag1d, us_10y_yield_change_lag1d, kosdaq150_5m_return_latest, krx300_5m_return_latest, korea_semiconductor_5m_return_latest |
| regular_0930 | 4 | kospi_5m_return_latest | korea_equity_market | 30.857 | spy_pre_event_return_5m, kospi_return_latest, usdkrw_return_latest, vix_level_lag1d, vix_change_lag1d, us_2y_yield_change_lag1d, us_10y_yield_change_lag1d, kosdaq150_5m_return_latest, krx300_5m_return_latest, korea_semiconductor_5m_return_latest |
