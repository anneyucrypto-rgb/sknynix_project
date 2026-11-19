# Premarket-to-open trading backtest

## Information and execution timeline

| Time (ET) | Event |
| --- | --- |
| Before 09:25 | Korean daily return, previous-close ECT, and both SOXX gauges are known |
| 09:25 | Freeze H2 forecast using no 09:30 information; enter in premarket |
| 09:25–09:27 | A protective stop-loss or take-profit may execute from one-minute high/low paths |
| Before 09:28 | Submit opposite-side MOO for any remaining position |
| 09:28–09:30 | No new protective exit is assumed after the MOO is locked |
| 09:30 | Exit remaining position at the opening-price proxy |

The H2 target is the full previous-close-to-open gap, but the strategy return is
only

\[
r^{trade}_d=\log\left(
\frac{P^{SKHY}_{d,09{:}30\ open}}
{P^{SKHY}_{d,09{:}25\ entry}}
\right).
\]

The backtest never credits the strategy with the already-realized portion of
the overnight gap.

## Step 1 — Expanding H2 gap forecast

For each date \(d\), estimate the retained SOXX-gauge H2 with dates strictly
before \(d\), then calculate the 09:25 gap nowcast \(\widehat r^{ON}_d\). H1
supplies the previous-close equilibrium error

\[
ECT_d=\log(P^{SKHY}_{d-1,close})
-\widehat c-\widehat\theta\log(P^{KR,USD}_d),
\]

while H2 combines it with the Korean daytime return and the SOXX
close-to-07:00 and 07:00-to-09:25 gauges. H2—not H1 directly—supplies the
return forecast.

## Step 2 — Prior-only calibration to remaining return

The first ten forecasts are calibration observations and remain flat. On every
later date, estimate with prior observations only:

\[
r^{trade}_t=a_d+b_d\widehat r^{ON}_t+u_t,
\qquad t<d.
\]

Then

\[
\widehat r^{trade}_d=\widehat a_d+\widehat b_d\widehat r^{ON}_d,
\qquad
\widehat\sigma_d=\sqrt{\frac{1}{N_d}\sum_{t<d}\widehat u_t^2}.
\]

This calibration is necessary because a model that predicts the full gap does
not automatically predict its final five-minute segment.

## Step 3 — Execution-cost hurdle

For candidate notional \(Q_d=|w_d|F_{d-1}\), model entry and exit impact with a
square-root participation rule:

\[
I_{d,s}=\iota\sqrt{\min(Q_d/V_{d,s},1)},
\]

where \(V_{d,s}\) is observed dollar volume at the entry or opening exit. The
round-trip cost hurdle is

\[
c_d=2(c^{transaction}+c^{slippage})+I_{d,entry}+I_{d,exit}+c^{buffer}.
\]

Trade only when

\[
|\widehat r^{trade}_d|>c_d.
\]

Thus buy when the calibrated expected return exceeds (c_d), short when it is
below (-c_d), and otherwise remain flat.

## Step 4 — Volatility-scaled position

\[
w_d=\operatorname{clip}\left(
k\frac{\widehat r^{trade}_d}{\widehat\sigma_d^2},
-W_{max},W_{max}
\right).
\]

The default cap is 100% gross exposure with no leverage. A quiet, high-conviction
forecast receives more weight than an equally sized forecast with high residual
volatility.

## Step 5 — Stop-loss and take-profit

The log-price distances are

\[
s_d=\max(s_{floor},m_s\widehat\sigma_d),
\qquad
g_d=\max(g_{floor},m_g\widehat\sigma_d).
\]

Defaults use \(m_s=0.75\), \(m_g=1.50\), a 25-bp stop floor, and a 50-bp
take-profit floor, creating a configured reward-to-risk ratio of approximately
2:1. One-minute bars from 09:25 through 09:27 determine whether either level is
hit. If both appear inside the same one-minute bar, the backtest assumes the
stop was hit first. This conservative convention avoids favorable intrabar
ordering assumptions.

If neither level is hit, an opposite-side MOO liquidates at the 09:30 opening
price proxy.

## Step 6 — Net return and fund

\[
r^{net}_d=w_dr^{asset}_d-|w_d|c^{execution}_d,
\qquad
F_d=F_{d-1}\exp(r^{net}_d).
\]

The fund begins at $10,000. Flat dates have zero return and zero cost.

## Step 7 — Drawdown controls

Let

\[
DD_d=F_d/\max_{t\le d}F_t-1.
\]

- reaching the pause threshold suspends new positions for the configured number
  of sessions;
- reaching the hard drawdown threshold or minimum fund stops future trading;
- trade-level stops remain independent of the portfolio pause.

## Reported statistics

The output includes total return, CAGR, annualized volatility, Sharpe, Sortino,
Calmar, maximum drawdown, directional accuracy, win rate, profit factor,
configured and realized risk-reward, average exposure, execution cost, stop and
take-profit counts, pause events, and ending fund.

The retained strategy passes its minimum economic screen only if it generates
at least three trades, positive total return, and positive Sharpe. The screen
does not select among models because the SOXX-gauge H2 is the only retained
forecast specification.

The current retained-model performance is written to
`design/TRADING_RESULTS.md` whenever `python -m backtest.strategy` completes.

## Auction-data caveat

The historical one-minute 09:30 open is used as an opening-auction proxy. It
does not establish that a particular MOO or LOO order received the official
opening price. Official Nasdaq opening-price, imbalance, order-submission, and
execution-report data are required for production-grade fill analysis.
