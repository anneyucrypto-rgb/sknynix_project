# Chronos-2 versus H1/H2 VIX forecast comparison

Both models forecast the daily SKHY previous-close-to-open log return on
the same dates. H2 re-estimates H1 and its ECT at every origin and uses
the fixed VIX-only p=q=0 specification. Chronos-2 uses the Korean
FX-adjusted return, VIX return, FLKR return, and USD/KRW return.

| Model | Forecasts | RMSE | MAE | Bias | Direction accuracy | OOS R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chronos-2 | 49 | 0.029204 | 0.022356 | 0.001770 | 73.47% | 0.442446 |
| h2-vix | 49 | 0.020878 | 0.017021 | 0.007863 | 83.67% | 0.715039 |

## Result

The VIX-only H1/H2 model has 28.5% lower RMSE and 23.9% lower MAE than Chronos-2, with a 10.20-percentage-point higher directional hit rate. On this short common-date sample, H1/H2 is the stronger forecasting model.
