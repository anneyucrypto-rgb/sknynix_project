# Granger ablation results summary

Generated: 2026-11-12 16:42:13 SGT

## Overall control-group influence

The table ranks groups using the absolute coefficient change

$$
\lvert \Delta \beta_0^g \rvert
= \lvert \widehat{\beta}_0^g
- \widehat{\beta}_0^{\mathrm{baseline}} \rvert .
$$

A larger change is sensitivity, not proof that the control is causal.

| Control group | Estimable cells | Mean abs change | Median abs change | Maximum abs change |
|---|---|---|---|---|
| korean_macro_only | 16 | 0.0883 | 0.0423 | 0.2264 |
| liquidity_only | 16 | 0.0029 | 0.0027 | 0.0071 |
| macro_only | 16 | 0.0927 | 0.0435 | 0.2397 |
| microstructure_only | 16 | 0.0044 | 0.0037 | 0.0154 |
| sector_only | 16 | 0.0091 | 0.0045 | 0.0238 |

The per-cell winner count identifies `macro_only` as the most frequent
source of Korean-coefficient movement in this run:

| Control group | Anchor/horizon wins |
|---|---|
| macro_only | 11 |
| korean_macro_only | 5 |

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
| premarket_dynamic | 5 | 0.0130 | 0.0598 | 0.0544 | 0.0004 | 0.0023 | 0.0011 | macro_only | p=0, q=0 | 80 |
| premarket_dynamic | 10 | 0.0085 | 0.0343 | 0.0260 | 0.0007 | 0.0031 | 0.0005 | macro_only | p=0, q=1 | 80 |
| premarket_dynamic | 15 | 0.0094 | 0.0479 | 0.0421 | 0.0007 | 0.0029 | 0.0006 | macro_only | p=0, q=1 | 80 |
| premarket_dynamic | 30 | -0.0034 | 0.0317 | 0.0233 | 0.0003 | 0.0014 | 0.0029 | macro_only | p=0, q=0 | 80 |
| premarket_dynamic | 45 | -0.0061 | 0.0367 | 0.0424 | 0.0019 | 0.0016 | 0.0023 | korean_macro_only | p=0, q=1 | 80 |
| premarket_dynamic | 60 | -0.0126 | 0.0277 | 0.0349 | 0.0018 | 0.0018 | 0.0040 | korean_macro_only | p=0, q=1 | 80 |
| premarket_dynamic | 100 | -0.0388 | 0.0146 | 0.0036 | 0.0038 | 0.0001 | 0.0038 | macro_only | p=0, q=0 | 80 |
| premarket_dynamic | 120 | -0.0544 | 0.0023 | 0.0109 | 0.0051 | 0.0026 | 0.0009 | korean_macro_only | p=0, q=1 | 80 |
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
| premarket_dynamic | 5 | 0.0130 | 0.0881 | 0.0881 | p=0, q=0 | 80 |
| premarket_dynamic | 10 | 0.0085 | 0.3050 | 0.1087 | p=0, q=1 | 80 |
| premarket_dynamic | 15 | 0.0094 | 0.2841 | 0.0816 | p=0, q=1 | 80 |
| premarket_dynamic | 30 | -0.0034 | 0.7833 | 0.7833 | p=0, q=0 | 80 |
| premarket_dynamic | 45 | -0.0061 | 0.6468 | 0.2334 | p=0, q=1 | 80 |
| premarket_dynamic | 60 | -0.0126 | 0.4780 | 0.1996 | p=0, q=1 | 80 |
| premarket_dynamic | 100 | -0.0388 | 0.1095 | 0.1095 | p=0, q=0 | 80 |
| premarket_dynamic | 120 | -0.0544 | 0.0218 | 0.0308 | p=0, q=1 | 80 |
| regular_0930 | 5 | 0.0038 | 0.8398 | 0.3009 | p=0, q=1 | 85 |
| regular_0930 | 10 | 0.0053 | 0.8819 | 0.8819 | p=0, q=0 | 85 |
| regular_0930 | 15 | -0.0120 | 0.8156 | 0.8156 | p=0, q=0 | 85 |
| regular_0930 | 30 | -0.0547 | 0.3226 | 0.3226 | p=0, q=0 | 85 |
| regular_0930 | 45 | -0.0832 | 0.2235 | 0.2235 | p=0, q=0 | 85 |
| regular_0930 | 60 | -0.0756 | 0.2767 | 0.2767 | p=0, q=0 | 85 |
| regular_0930 | 100 | -0.0599 | 0.4217 | 0.4217 | p=0, q=0 | 85 |
| regular_0930 | 120 | -0.0779 | 0.2802 | 0.2802 | p=0, q=0 | 85 |

At `premarket_dynamic`, the current Korean coefficient is significant at 5% for the 120-minute responses; the selected Korean block is jointly significant for the 120-minute responses.

At `regular_0930`, the current Korean coefficient is significant at 5% for none of the responses; the selected Korean block is jointly significant for none of the responses.

## Availability

Unavailable cells: None.

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
