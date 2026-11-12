# Granger ablation results summary

Generated: 2026-11-11 18:38:39 SGT

## Overall control-group influence

The table ranks groups using the absolute change in the current Korean-return
coefficient relative to the no-group baseline. A larger change is sensitivity,
not proof that the control is causal.

| Control group | Estimable cells | Mean abs change | Median abs change | Maximum abs change |
|---|---|---|---|---|
| liquidity_only | 6 | 0.0042 | 0.0044 | 0.0063 |
| macro_only | 12 | 0.1176 | 0.1089 | 0.2416 |
| microstructure_only | 6 | 0.0064 | 0.0058 | 0.0126 |
| sector_only | 12 | 0.0071 | 0.0020 | 0.0201 |

The per-cell winner count identifies `macro_only` as the most frequent
source of Korean-coefficient movement in this run:

| Control group | Anchor/horizon wins |
|---|---|
| macro_only | 12 |

The macro block contains seven controls in a sample of roughly 80 event days.
Its coefficient shifts can therefore reflect common-factor adjustment, but
also multicollinearity or over-parameterization. Treat the ranking as a
sensitivity diagnostic rather than evidence that the macro block is causal.

Because liquidity is estimable only at 09:30, the like-for-like regular-session
comparison is:

| Control group | Regular cells | Mean abs change | Median abs change | Maximum abs change |
|---|---|---|---|---|
| liquidity_only | 6 | 0.0042 | 0.0044 | 0.0063 |
| macro_only | 6 | 0.1365 | 0.1484 | 0.2416 |
| microstructure_only | 6 | 0.0064 | 0.0058 | 0.0126 |
| sector_only | 6 | 0.0137 | 0.0177 | 0.0201 |

## Anchor-by-horizon comparison

`Macro`, `sector`, and `liquidity` columns report
`abs(beta_group - beta_baseline)`.

| Anchor | Horizon | Baseline beta KR | Macro | Sector | Liquidity | Microstructure | Largest change | Selected order | N |
|---|---|---|---|---|---|---|---|---|---|
| premarket_0400 | 5 | -0.0220 | 0.1059 | 0.0002 | — | — | macro_only | p=0, q=0 | 85 |
| premarket_0400 | 10 | -0.0167 | 0.0775 | 0.0000 | — | — | macro_only | p=0, q=2 | 85 |
| premarket_0400 | 30 | -0.0067 | 0.1118 | 0.0016 | — | — | macro_only | p=1, q=0 | 85 |
| premarket_0400 | 60 | -0.0355 | 0.1159 | 0.0012 | — | — | macro_only | p=0, q=0 | 85 |
| premarket_0400 | 90 | -0.0626 | 0.0786 | 0.0005 | — | — | macro_only | p=1, q=0 | 82 |
| premarket_0400 | 120 | -0.0691 | 0.1018 | 0.0002 | — | — | macro_only | p=1, q=0 | 81 |
| regular_0930 | 5 | 0.0115 | 0.0057 | 0.0024 | 0.0044 | 0.0039 | macro_only | p=0, q=0 | 84 |
| regular_0930 | 10 | 0.0061 | 0.0401 | 0.0048 | 0.0045 | 0.0067 | macro_only | p=0, q=0 | 84 |
| regular_0930 | 30 | -0.0558 | 0.2416 | 0.0183 | 0.0034 | 0.0030 | macro_only | p=0, q=0 | 84 |
| regular_0930 | 60 | -0.0822 | 0.2351 | 0.0201 | 0.0019 | 0.0049 | macro_only | p=0, q=0 | 84 |
| regular_0930 | 90 | -0.0724 | 0.1242 | 0.0192 | 0.0063 | 0.0072 | macro_only | p=0, q=0 | 84 |
| regular_0930 | 120 | -0.0891 | 0.1726 | 0.0172 | 0.0044 | 0.0126 | macro_only | p=0, q=0 | 84 |

## Baseline Granger inference

`Beta p` is the HAC p-value for the current Korean-return coefficient.
`Block p` is the HAC joint Wald p-value for all selected Korean-return terms.

| Anchor | Horizon | Beta KR | Beta p | Block p | Selected order | N |
|---|---|---|---|---|---|---|
| premarket_0400 | 5 | -0.0220 | 0.1350 | 0.1350 | p=0, q=0 | 85 |
| premarket_0400 | 10 | -0.0167 | 0.2909 | 0.0007 | p=0, q=2 | 85 |
| premarket_0400 | 30 | -0.0067 | 0.7335 | 0.7335 | p=1, q=0 | 85 |
| premarket_0400 | 60 | -0.0355 | 0.0349 | 0.0349 | p=0, q=0 | 85 |
| premarket_0400 | 90 | -0.0626 | 0.0005 | 0.0005 | p=1, q=0 | 82 |
| premarket_0400 | 120 | -0.0691 | 0.0130 | 0.0130 | p=1, q=0 | 81 |
| regular_0930 | 5 | 0.0115 | 0.6099 | 0.6099 | p=0, q=0 | 84 |
| regular_0930 | 10 | 0.0061 | 0.8674 | 0.8674 | p=0, q=0 | 84 |
| regular_0930 | 30 | -0.0558 | 0.3184 | 0.3184 | p=0, q=0 | 84 |
| regular_0930 | 60 | -0.0822 | 0.2366 | 0.2366 | p=0, q=0 | 84 |
| regular_0930 | 90 | -0.0724 | 0.3271 | 0.3271 | p=0, q=0 | 84 |
| regular_0930 | 120 | -0.0891 | 0.2117 | 0.2117 | p=0, q=0 | 84 |

At the 04:00 anchor, the current Korean coefficient is significant at 5% for
the 60-, 90-, and 120-minute baseline responses. The 10-minute baseline selects
`q=2`: its current-return coefficient is not individually significant, but the
Korean block is jointly significant. No 09:30 baseline Korean coefficient or
joint block is significant at 5%.

## Availability

Unavailable cells: premarket_0400/5m/liquidity_only, premarket_0400/5m/microstructure_only, premarket_0400/10m/liquidity_only, premarket_0400/10m/microstructure_only, premarket_0400/30m/liquidity_only, premarket_0400/30m/microstructure_only, premarket_0400/60m/liquidity_only, premarket_0400/60m/microstructure_only, premarket_0400/90m/liquidity_only, premarket_0400/90m/microstructure_only, premarket_0400/120m/liquidity_only, premarket_0400/120m/microstructure_only.

Liquidity-only and microstructure-only models at 04:00 are unavailable because
the source panel has no complete 30-minute pre-event window at the start of
premarket.

## Interpretation limits

- All comparisons use a common sample and identical selected lag orders within
  an anchor/horizon cell.
- HAC inference is reported in the machine-readable results, but coefficient
  movement alone does not establish significance or causal mediation.
- The six horizons overlap, the sample is small, and no multiple-testing
  correction is applied in this descriptive ablation ranking.
- Post-anchor trade/quote variables are excluded to prevent look-ahead bias.

Full estimates are stored only in
`models/results/granger_ablation_results.parquet`.
