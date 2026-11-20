# Premarket-to-open forecast strategy results

Each forecast file was evaluated independently under the 09:25–09:30 ET rules.

## Close-to-open forecast accuracy

RMSE and MAE are in basis points of log return. Directional accuracy is the
share of evaluated sessions where the forecast and realized return have the same sign.

| Model | Sessions | RMSE (bps) | MAE (bps) | Directional accuracy |
| --- | ---: | ---: | ---: | ---: |
| h2 | 48 | 130.20 | 103.13 | 97.92% |
| chronos-2 | 48 | 188.74 | 135.97 | 93.75% |

## Trading performance

Stop-loss exits are disabled. Take-profit, execution cost, sizing, and
portfolio risk settings are the same as in the stop-loss baseline.

| Model | Evaluated sessions | Excluded | Trades | Net return | Sharpe | Ending fund |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| h2 | 2026-09-10–2026-11-16 (48) | 0 | 4 | -0.57% | -1.89 | $9,942.52 |
| chronos-2 | 2026-09-10–2026-11-16 (48) | 0 | 4 | -0.66% | -2.10 | $9,934.36 |

## Trade breakdown

Forecast accuracy measures the prior close to the 09:30 open. Trading P&L
starts at the 09:25 entry and reflects the calibrated signal, position size,
exit price, and execution costs. Gross P&L uses each trade’s starting fund
before costs. Net P&L is the actual change in the simulated fund.

### h2

| Session | Side | Exposure | Entry | Exit | Exit reason | Gross P&L (USD) | Cost (USD) | Net P&L (USD) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 2026-10-19 | Short | 46.82% | $143.4900 | $143.2450 | 09:30 opening auction | +8.00 | 3.91 | +4.09 |
| 2026-10-28 | Short | 57.04% | $155.9900 | $157.7600 | 09:30 opening auction | -64.17 | 4.61 | -68.78 |
| 2026-11-06 | Short | 31.26% | $129.6100 | $129.1250 | 09:30 opening auction | +11.65 | 2.55 | +9.10 |
| 2026-11-11 | Short | 32.80% | $139.6600 | $139.6250 | 09:30 opening auction | +0.82 | 2.71 | -1.89 |

Gross gains were $20.47 against $64.17 in gross losses. Execution costs reduced the result by $13.78, leaving net P&L of -57.48 USD.

### chronos-2

| Session | Side | Exposure | Entry | Exit | Exit reason | Gross P&L (USD) | Cost (USD) | Net P&L (USD) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 2026-09-24 | Long | 33.78% | $185.8000 | $185.5700 | 09:30 opening auction | -4.18 | 2.84 | -7.03 |
| 2026-10-19 | Short | 38.29% | $143.4900 | $143.2450 | 09:30 opening auction | +6.54 | 3.18 | +3.36 |
| 2026-10-28 | Short | 58.51% | $155.9900 | $157.7600 | 09:30 opening auction | -65.78 | 4.72 | -70.50 |
| 2026-11-06 | Short | 29.32% | $129.6100 | $129.1250 | 09:30 opening auction | +10.92 | 2.39 | +8.53 |

Gross gains were $17.46 against $69.96 in gross losses. Execution costs reduced the result by $13.14, leaving a net P&L of -65.64 USD.

## Previously stopped trades at the 09:30 exit

The auction scenario uses the no-stop rerun’s position size and execution
cost rate. If a take-profit exits earlier, the rerun P&L reflects that exit.

| Model | Session | 09:30 open | Original stop P&L (USD) | 09:30 P&L (USD) | Change (USD) | Rerun exit | Rerun P&L (USD) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| h2 | 2026-10-28 | $157.7600 | -18.89 | -68.78 | -49.90 | 09:30 opening auction | -68.78 |
| h2 | 2026-11-06 | $129.1250 | -11.75 | +9.10 | +20.85 | 09:30 opening auction | +9.10 |
| chronos-2 | 2026-10-28 | $157.7600 | -19.40 | -70.50 | -51.10 | 09:30 opening auction | -70.50 |
| chronos-2 | 2026-11-06 | $129.1250 | -11.06 | +8.53 | +19.59 | 09:30 opening auction | +8.53 |
