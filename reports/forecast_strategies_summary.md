# Premarket-to-open forecast strategy results

Both common-date forecast files were evaluated independently under the same 09:25–09:30 ET rules.

## Close-to-open forecast accuracy

RMSE and MAE are in basis points of log return. Directional accuracy is the
share of evaluated sessions where the forecast and realized return have the same sign.

| Model | Sessions | RMSE (bps) | MAE (bps) | Directional accuracy |
| --- | ---: | ---: | ---: | ---: |
| h2-vix | 49 | 208.78 | 170.21 | 83.67% |
| chronos-2 | 49 | 292.04 | 223.56 | 73.47% |

## Trading performance

| Model | Evaluated sessions | Excluded | Trades | Net return | Sharpe | Ending fund |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| h2-vix | 2026-09-09–2026-11-16 (49) | 0 | 2 | -0.12% | -3.00 | $9,988.24 |
| chronos-2 | 2026-09-09–2026-11-16 (49) | 0 | 3 | -0.31% | -2.92 | $9,968.78 |

**Conclusion:** Neither model passes the minimum screen of at least three trades, positive net return, and positive Sharpe.

## Trade breakdown

Forecast accuracy measures the prior close to the 09:30 open. Trading P&L
starts at the 09:25 entry and reflects the calibrated signal, position size,
exit price, and execution costs. Gross P&L uses each trade’s starting fund
before costs. Net P&L is the actual change in the simulated fund.

### h2-vix

| Session | Side | Exposure | Entry | Exit | Exit reason | Gross P&L (USD) | Cost (USD) | Net P&L (USD) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 2026-09-24 | Long | 39.54% | $185.8000 | $185.5700 | 09:30 opening auction | -4.90 | 3.34 | -8.24 |
| 2026-09-25 | Long | 27.87% | $189.4400 | $189.3600 | 09:30 opening auction | -1.18 | 2.34 | -3.51 |

Gross gains were $0.00 against $6.07 in gross losses. Execution costs reduced the result by $5.68, leaving net P&L of -11.76 USD.

### chronos-2

| Session | Side | Exposure | Entry | Exit | Exit reason | Gross P&L (USD) | Cost (USD) | Net P&L (USD) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 2026-09-23 | Long | 52.57% | $191.3300 | $191.6000 | 09:30 opening auction | +7.42 | 4.38 | +3.04 |
| 2026-09-24 | Long | 81.66% | $185.8000 | $185.5700 | 09:30 opening auction | -10.11 | 7.07 | -17.18 |
| 2026-10-15 | Short | 49.23% | $137.3000 | $137.6653 | Stop-loss at 09:25 | -13.05 | 4.03 | -17.08 |

Gross gains were $7.42 against $23.17 in gross losses, including $13.05 across 1 stop-loss exits. Execution costs reduced the result by $15.47, leaving net P&L of -31.22 USD.
