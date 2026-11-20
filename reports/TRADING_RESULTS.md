# Premarket-to-open trading results

**Run ID:** `overnight-best`  
**Generated:** 2026-11-19T05:41:51.999980+00:00  
**Forecast model:** retained bivariate-H1/SOXX-gauge H2  
**Trading interval:** 09:25 premarket entry to 09:30 opening auction

## Forecast input

The strategy uses H2 forecasts. H1 supplies only the equilibrium benchmark used inside H2.

| Forecast measure | Result |
| --- | ---: |
| H2 specification | `korean_ect_soxx_gauges` |
| Complete observations | 88 |
| Expanding-window forecasts | 48 |
| RMSE | 0.012906 |
| MAE | 0.010055 |
| Forecast directional accuracy | 95.8% |
| OOS R² versus zero | 0.8930 |
| H2 adjustment coefficient | -0.226168 |
| H2 adjustment p-value | 8.63e-07 |

## Execution rules

1. Freeze the H2 forecast at 09:25 ET and calibrate it to the remaining 09:25–09:30 return using prior observations only.
2. Enter at the 09:25 SKHY premarket open only when expected profit exceeds transaction costs, slippage, impact, and the safety buffer.
3. Keep volatility-scaled protective stop-loss and take-profit orders active through 09:27.
4. Submit an opposite-side MOO before 09:28 for every survivor and exit at 09:30.
5. Apply volatility-scaled sizing, drawdown pauses, the hard drawdown stop, and the minimum-fund floor.

Only the executable 09:25–09:30 return is credited. The already-realized portion of the full overnight gap is excluded.

## Cost and risk assumptions

| Assumption | Value |
| --- | ---: |
| Safety buffer | 10.00 bps |
| Transaction cost | 1.00 bps per side |
| Slippage | 3.00 bps per side |
| Impact coefficient | 10.00 bps |
| Maximum weight | 100.0% |
| Stop / take-profit volatility multipliers | 0.75 / 1.50 |
| Drawdown pause threshold | 1.00% |
| Hard drawdown stop | 5.00% |

## Trading performance

| Measure | Result |
| --- | ---: |
| Forecast observations | 48 |
| Active trades | 4 |
| Long / short / flat | 0 / 4 / 44 |
| Stop / take-profit / MOO exits | 2 / 0 / 2 |
| Traded directional accuracy | 50.0% |
| Win rate | 25.0% |
| Net cumulative return | -0.29% |
| CAGR | -1.52% |
| Sharpe ratio | -2.9140 |
| Sortino ratio | -2.9415 |
| Calmar ratio | -4.5799 |
| Maximum drawdown | -0.33% |
| Realized risk-to-reward | 0.367 |
| Profit factor | 0.122 |
| Average execution cost | 8.25 bps |
| Initial fund | $10,000.00 |
| Remaining fund | $9,970.93 |

**Conclusion:** The strategy does not pass the minimum economic screen of at least three trades, positive net return, and positive Sharpe.

Nasdaq's MOO timing is documented in the [Opening/Closing Cross FAQ](https://beta.nasdaqtrader.com/content/ProductsServices/Trading/Crosses/openclose_faqs.pdf). The one-minute 09:30 open remains an auction-price proxy rather than an order-level fill record.
