# Granger ablation results summary

Generated: 2026-11-12 15:50:51 SGT

## Overall control-group influence

The table ranks groups using the absolute coefficient change

$$
\left|\Delta \beta_0^g\right|
= \left|
\widehat{\beta}_0^g
- \widehat{\beta}_0^{\mathrm{baseline}}
\right| .
$$

A larger change is sensitivity, not proof that the control is causal.

| Control group | Estimable cells | Mean abs change | Median abs change | Maximum abs change |
|---|---|---|---|---|
| korean_macro_only | 8 | 0.1469 | 0.1685 | 0.2264 |
| liquidity_only | 8 | 0.0039 | 0.0040 | 0.0071 |
| macro_only | 8 | 0.1535 | 0.1683 | 0.2397 |
| microstructure_only | 8 | 0.0068 | 0.0064 | 0.0154 |
| sector_only | 8 | 0.0163 | 0.0208 | 0.0238 |

The per-cell winner count identifies `macro_only` as the most frequent
source of Korean-coefficient movement in this run:

| Control group | Anchor/horizon wins |
|---|---|
| macro_only | 6 |
| korean_macro_only | 2 |

The macro block contains seven controls in a sample of roughly 80 event days.
Its coefficient shifts can therefore reflect common-factor adjustment, but
also multicollinearity or over-parameterization. Treat the ranking as a
sensitivity diagnostic rather than evidence that the macro block is causal.

The regular-session-only comparison is:

| Control group | Regular cells | Mean abs change | Median abs change | Maximum abs change |
|---|---|---|---|---|
| korean_macro_only | 8 | 0.1469 | 0.1685 | 0.2264 |
| liquidity_only | 8 | 0.0039 | 0.0040 | 0.0071 |
| macro_only | 8 | 0.1535 | 0.1683 | 0.2397 |
| microstructure_only | 8 | 0.0068 | 0.0064 | 0.0154 |
| sector_only | 8 | 0.0163 | 0.0208 | 0.0238 |

## Anchor-by-horizon comparison

`Macro`, `Korean macro`, `sector`, `liquidity`, and `microstructure` columns
report the absolute coefficient change defined above.

| Anchor | Horizon | Baseline beta KR | Macro | Korean macro | Sector | Liquidity | Microstructure | Largest change | Selected order | N |
|---|---|---|---|---|---|---|---|---|---|---|
| regular_0930 | 5 | 0.0038 | 0.0111 | 0.0155 | 0.0031 | 0.0054 | 0.0047 | korean_macro_only | p=0, q=1 | 85 |
| regular_0930 | 10 | 0.0053 | 0.0391 | 0.0303 | 0.0052 | 0.0042 | 0.0066 | macro_only | p=0, q=0 | 85 |
| regular_0930 | 15 | -0.0120 | 0.1539 | 0.1371 | 0.0101 | 0.0071 | 0.0061 | macro_only | p=0, q=0 | 85 |
| regular_0930 | 30 | -0.0547 | 0.2397 | 0.2264 | 0.0205 | 0.0036 | 0.0029 | macro_only | p=0, q=0 | 85 |
| regular_0930 | 45 | -0.0832 | 0.2219 | 0.2086 | 0.0232 | 0.0004 | 0.0037 | macro_only | p=0, q=0 | 85 |
| regular_0930 | 60 | -0.0756 | 0.2257 | 0.2206 | 0.0236 | 0.0021 | 0.0068 | macro_only | p=0, q=0 | 85 |
| regular_0930 | 100 | -0.0599 | 0.1771 | 0.1764 | 0.0238 | 0.0038 | 0.0086 | macro_only | p=0, q=0 | 85 |
| regular_0930 | 120 | -0.0779 | 0.1596 | 0.1605 | 0.0211 | 0.0042 | 0.0154 | korean_macro_only | p=0, q=0 | 85 |

## Baseline Granger inference

`Beta KR` is $\widehat{\beta}_0$, the estimated change in the future SKHY
log return associated with a one-unit increase in `kr_return_0`, conditional on
the selected lags and controls. Its sign gives the direction of the association.
`Beta p` is the two-sided HAC-robust p-value for
$H_0:\beta_0=0$; it measures evidence against a zero current-Korean-return
coefficient, not the economic size of the coefficient. `Block p` is the
HAC-robust joint Wald p-value for all Korean-return terms retained by BIC.

| Anchor | Horizon | Beta KR | Beta p | Block p | Selected order | N |
|---|---|---|---|---|---|---|
| regular_0930 | 5 | 0.0038 | 0.8398 | 0.3009 | p=0, q=1 | 85 |
| regular_0930 | 10 | 0.0053 | 0.8819 | 0.8819 | p=0, q=0 | 85 |
| regular_0930 | 15 | -0.0120 | 0.8156 | 0.8156 | p=0, q=0 | 85 |
| regular_0930 | 30 | -0.0547 | 0.3226 | 0.3226 | p=0, q=0 | 85 |
| regular_0930 | 45 | -0.0832 | 0.2235 | 0.2235 | p=0, q=0 | 85 |
| regular_0930 | 60 | -0.0756 | 0.2767 | 0.2767 | p=0, q=0 | 85 |
| regular_0930 | 100 | -0.0599 | 0.4217 | 0.4217 | p=0, q=0 | 85 |
| regular_0930 | 120 | -0.0779 | 0.2802 | 0.2802 | p=0, q=0 | 85 |

At `premarket_dynamic`, no baseline model is estimable in this run.

At `regular_0930`, the current Korean coefficient is significant at 5% for none of the responses; the selected Korean block is jointly significant for none of the responses.

## Availability

Unavailable cells: premarket_dynamic/5m/baseline, premarket_dynamic/5m/korean_macro_only, premarket_dynamic/5m/liquidity_only, premarket_dynamic/5m/macro_only, premarket_dynamic/5m/microstructure_only, premarket_dynamic/5m/sector_only, premarket_dynamic/10m/baseline, premarket_dynamic/10m/korean_macro_only, premarket_dynamic/10m/liquidity_only, premarket_dynamic/10m/macro_only, premarket_dynamic/10m/microstructure_only, premarket_dynamic/10m/sector_only, premarket_dynamic/15m/baseline, premarket_dynamic/15m/korean_macro_only, premarket_dynamic/15m/liquidity_only, premarket_dynamic/15m/macro_only, premarket_dynamic/15m/microstructure_only, premarket_dynamic/15m/sector_only, premarket_dynamic/30m/baseline, premarket_dynamic/30m/korean_macro_only, premarket_dynamic/30m/liquidity_only, premarket_dynamic/30m/macro_only, premarket_dynamic/30m/microstructure_only, premarket_dynamic/30m/sector_only, premarket_dynamic/45m/baseline, premarket_dynamic/45m/korean_macro_only, premarket_dynamic/45m/liquidity_only, premarket_dynamic/45m/macro_only, premarket_dynamic/45m/microstructure_only, premarket_dynamic/45m/sector_only, premarket_dynamic/60m/baseline, premarket_dynamic/60m/korean_macro_only, premarket_dynamic/60m/liquidity_only, premarket_dynamic/60m/macro_only, premarket_dynamic/60m/microstructure_only, premarket_dynamic/60m/sector_only, premarket_dynamic/100m/baseline, premarket_dynamic/100m/korean_macro_only, premarket_dynamic/100m/liquidity_only, premarket_dynamic/100m/macro_only, premarket_dynamic/100m/microstructure_only, premarket_dynamic/100m/sector_only, premarket_dynamic/120m/baseline, premarket_dynamic/120m/korean_macro_only, premarket_dynamic/120m/liquidity_only, premarket_dynamic/120m/macro_only, premarket_dynamic/120m/microstructure_only, premarket_dynamic/120m/sector_only.

Dynamic-premarket cells are unavailable when full quote ticks are absent, no
candidate passes the predeclared liquidity filters, or the required model
variables are incomplete. No unavailable anchor is time- or price-filled.

## Interpretation limits

- All comparisons use a common sample and identical selected lag orders within
  an anchor/horizon cell.
- HAC inference is reported in the machine-readable results, but coefficient
  movement alone does not establish significance or causal mediation.
- The eight horizons overlap, the sample is small, and no multiple-testing
  correction is applied in this descriptive ablation ranking.
- Post-anchor trade/quote variables are excluded to prevent look-ahead bias.

Full estimates are stored only in
`models/results/granger_ablation_results.parquet`.
