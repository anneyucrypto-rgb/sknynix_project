For your Korea–U.S. ADR price-discovery research, I recommend extending the existing SKHY event-time ARDL framework into a sector-specific, multi-layer control system.

The key is to distinguish three sources of price movements:

1. Macroeconomic factors: Exchange rates, interest rates, inflation, global risk sentiment, and commodity prices.
2. Industry and company fundamentals (microeconomic factors): Display-panel prices, telecom competition, steel demand, bank profitability, electricity tariffs, and other company-specific drivers.
3. Market microstructure: Bid–ask spreads, trading volume, liquidity, volatility, and the speed at which information enters U.S. ADR prices.

These should not all be included indiscriminately. With your event-time ARDL, the objective is to isolate the incremental predictive contribution of Korean stock returns while avoiding overfitting and look-ahead bias.

## 1. Recommended control architecture

I would organize controls into five groups.

| Group                     | Controls                                      | Purpose                                                         |
| ------------------------- | --------------------------------------------- | --------------------------------------------------------------- |
| C1: Global macro          | SPY, VIX, U.S. Treasury yields                | General U.S. market and risk conditions                         |
| C2: Korea macro           | KOSPI, USD/KRW, Korean rates                  | Korean-market conditions and currency transmission              |
| C3: Sector                | Sector ETF/index, relevant commodities        | Separate industry-wide shocks from company-specific information |
| C4: Company fundamentals  | Earnings, financial ratios, regulatory events | Capture firm-specific economic developments                     |
| C5: Market microstructure | ADR volume, spread, volatility, price gap     | Account for trading frictions and price-discovery speed         |

For a predictive model, each feature must be observable before the forecast starts. For a total-response model, controls measured after the Korean close may already reflect the Korean information and should be treated separately as potential mediators.

## 2. Sector-specific macroeconomic and microeconomic controls

The following are recommended candidate variables, not all mandatory regressors. I distinguish high-frequency trading controls from slower fundamental variables.

### A. LG Display — 034220.KS / LPL

Sector: OLED, LCD, consumer electronics

| Control                     | Variable/proxy                    | Priority              |
| --------------------------- | --------------------------------- | --------------------- |
| U.S. technology market      | QQQ return                        | High                  |
| Semiconductor/electronics   | SOXX return                       | Medium                |
| Display-panel prices        | OLED/LCD panel ASP index          | High for fundamentals |
| Major customer demand       | Apple (AAPL) return               | Medium                |
| Consumer electronics demand | PC/smartphone shipment indicators | Medium                |
| Exchange rate               | USD/KRW                           | High                  |
| Component input costs       | Memory/component price indices    | Medium                |

LG Display's recent filings confirm that product mix and panel selling prices materially influence its business, with significant exposure to IT and mobile displays.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.sec.gov\&sz=32)

SEC+1



Suggested daily model: QQQ + USD/KRW + one electronics-sector factor. Use panel ASP and shipment data for lower-frequency research, not as fabricated five-minute signals.

### B. SK Telecom and KT — 017670.KS / SKM, 030200.KS / KT

Sector: Telecommunications

| Control                      | Variable/proxy                            | Priority              |
| ---------------------------- | ----------------------------------------- | --------------------- |
| U.S. telecom sector          | IYZ ETF                                   | High                  |
| U.S. market                  | SPY                                       | High                  |
| Korean interest rates        | Korea 3-year government bond yield        | High                  |
| U.S. interest rates          | U.S. 10-year Treasury yield               | Medium                |
| Domestic telecom competition | Subscriber market share, churn            | Medium                |
| Operating performance        | ARPU, subscriber growth                   | High for fundamentals |
| Regulation                   | Spectrum auctions, telecom pricing policy | Event control         |
| FX                           | USD/KRW                                   | High                  |

Telecom operators are relatively defensive businesses, but their valuations can still be sensitive to financing costs, dividends, competition, and regulation.

Suggested daily model: IYZ + SPY + USD/KRW, with Korean yields and regulatory-event indicators as extensions.

For comparing SKM and KT, a Korean telecom-sector index would be particularly useful. It can help separate common Korean telecom shocks from firm-specific information.

### C. POSCO Holdings — 005490.KS / PKX

Sector: Steel, industrial materials, battery materials

| Control                    | Variable/proxy                       | Priority    |
| -------------------------- | ------------------------------------ | ----------- |
| U.S. steel sector          | SLX ETF                              | High        |
| Iron ore prices            | Iron ore futures/index               | High        |
| Steel prices               | HRC steel futures                    | High        |
| China industrial demand    | China steel PMI, property indicators | High        |
| Battery-material prices    | Lithium carbonate prices             | Medium–High |
| Global industrial activity | Copper futures                       | Medium      |
| FX                         | USD/KRW, USD/CNY                     | High        |
| Broad market               | SPY                                  | Medium      |

Suggested daily model: SLX + iron ore + USD/KRW + China-related market factor.

POSCO is especially interesting because it is exposed to both steel and battery-material economics. These factors may influence the Korean close and subsequent U.S. ADR repricing.

For high-frequency modeling, distinguish U.S. steel futures prices available at the anchor from Asian commodity prices that may have stopped updating hours earlier.

### D. KB Financial, Shinhan Financial, Woori Financial

105560.KS / KB, 055550.KS / SHG, 316140.KS / WF

These three companies can share a substantial portion of their control architecture.

| Control                | Variable/proxy                       | Priority              |
| ---------------------- | ------------------------------------ | --------------------- |
| U.S. banking sector    | KBE or KRE ETF                       | High                  |
| U.S. financial sector  | XLF ETF                              | High                  |
| Korean interest rates  | Korea 3-year and 10-year yields      | High                  |
| Yield-curve slope      | Korea 10Y − 3Y                       | High                  |
| Credit conditions      | Korean corporate credit spread       | High                  |
| Loan-market conditions | Household credit growth              | Medium                |
| Bank profitability     | NIM, NPL ratio, loan-loss provisions | High for fundamentals |
| Real estate risk       | Korean housing price index           | Medium                |
| FX                     | USD/KRW                              | High                  |

Korean bank lending restrictions and interest-rate spreads are economically relevant: recent reporting documents changes in household loan margins associated with tighter lending limits.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://biz.chosun.com\&sz=32)

CHOSUNBIZ+1



Suggested daily model: XLF + Korean yield changes + USD/KRW + a Korean bank-sector index.

For high-frequency trading, the key financial-market variables are bank-sector equity returns, yield changes, FX, and credit-risk proxies. NIM, NPL, and lending growth are better suited to quarterly or monthly fundamental analysis.

### E. Korea Electric Power — 015760.KS / KEP

Sector: Regulated electricity utility

| Control                     | Variable/proxy                          | Priority |
| --------------------------- | --------------------------------------- | -------- |
| U.S. utilities sector       | XLU ETF                                 | High     |
| Natural gas                 | LNG benchmark (JKM), or Henry Hub proxy | High     |
| Coal                        | Newcastle thermal coal benchmark        | High     |
| Crude oil                   | Brent futures                           | Medium   |
| Korean interest rates       | Korea government bond yields            | High     |
| Electricity tariffs         | Announced tariff changes                | High     |
| Domestic electricity demand | Industrial output, temperature          | Medium   |
| FX                          | USD/KRW                                 | High     |

KEPCO is especially sensitive to the relationship between regulated electricity tariffs and fuel-procurement costs. Its 2026 fourth-quarter tariff decision illustrates why regulation and fuel prices should be modeled together.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://en.yna.co.kr\&sz=32)

Yonhap News Agency+1



Suggested daily model: XLU + LNG/coal prices + USD/KRW + tariff-announcement indicators.

Note that Henry Hub gas prices are not a direct substitute for Asian LNG import costs; use the JKM benchmark where suitable data are available.

## 3. Common controls across all eight ADRs

Rather than building eight unrelated models, I recommend a shared feature table plus sector-specific extensions.

| Variable                   | Frequency         | Function                         |
| -------------------------- | ----------------- | -------------------------------- |
| SPY return                 | 5-minute          | Broad U.S. equity market         |
| KOSPI return               | Daily             | Korean equity-market conditions  |
| USD/KRW return             | 5-minute          | Currency transmission            |
| VIX level/change           | Daily or intraday | Risk sentiment                   |
| U.S. 2Y/10Y yield changes  | Intraday/daily    | Monetary conditions              |
| Korea 3Y/10Y yield changes | Daily             | Domestic discount rates          |
| Sector ETF return          | 5-minute          | U.S. industry-wide movements     |
| Korean sector index return | Daily             | Domestic industry-wide movements |
| Macro announcement flags   | Event-based       | Scheduled information shocks     |

The KOSPI is particularly important for your research because a Korean company's daily return may partly reflect a broad Korean-market rally rather than company-specific price discovery.

One specification is to replace raw Korean returns with market-adjusted Korean returns:

\\[ r^{KR,abnormal}\_{i,d} =r^{KR}\_{i,d} -\hat b_i r^{KOSPI}\_d. \\]

Here \\(\hat b_i\\) is estimated using historical data available before day \\(d\\). This provides a robustness test of whether company-specific Korean information predicts U.S. ADR movements.

However, this is not necessarily superior to including the raw Korean return and KOSPI separately. Both specifications should be tested.

## 4. Market microstructure controls

These are different from company fundamentals. They are particularly important for ADRs because the U.S. instruments may have relatively low premarket liquidity.

| Control                  | Mathematical definition                   | Purpose                   |
| ------------------------ | ----------------------------------------- | ------------------------- |
| Relative spread          | \\((Ask-Bid)/Mid\\)                       | Transaction cost          |
| Order-book imbalance     | \\((BidSize-AskSize)/(BidSize+AskSize)\\) | Short-term order pressure |
| Realized volatility      | \\(\sum_k r_k^2\\)                        | Recent price variability  |
| Relative volume          | Volume / historical time-of-day volume    | Abnormal trading activity |
| Price-to-VWAP deviation  | \\(\ln(P/VWAP)\\)                         | Short-term price pressure |
| Trade count              | Number of transactions                    | Trading intensity         |
| Staleness                | Time since last valid trade               | Price reliability         |
| ADR fair-value deviation | \\(\ln(P\_{ADR}/P\_{fair})\\)             | Cross-market mispricing   |

For these eight ADRs, I would pay special attention to staleness and bid–ask spreads, because an apparent arbitrage opportunity can simply reflect a U.S. price that has not traded recently.

At 04:00 ET, use quote snapshots and opening-liquidity indicators if available. Do not construct a preceding 30-minute same-day return when no trading history exists.

## 5. How to incorporate these controls into event-time ARDL

For company \\(i\\), event day \\(d\\), and forecast horizon \\(h\\), extend your model to:

\\[ \boxed{ \begin{aligned} r^{US,h}\_{i,d}={}&\alpha\_{i,h} +\sum\_{\ell=1}^{p}\phi\_{\ell,i,h}r^{US,h}\_{i,d-\ell}\\\ &+\sum\_{j=0}^{q}\beta\_{j,i,h}r^{KR}\_{i,d-j}\\\ &+\gamma\_{i,h}^{\top}M_d +\delta\_{i,h}^{\top}S\_{i,d}\\\ &+\eta\_{i,h}^{\top}L\_{i,d} +\varepsilon\_{i,d,h}. \end{aligned} } \\]

Where:

- \\(M_d\\): Common macroeconomic controls.
- \\(S\_{i,d}\\): Sector-specific market and fundamental controls.
- \\(L\_{i,d}\\): ADR microstructure controls known before the prediction window.

The key hypothesis remains:

\\[ H_0:\beta\_{0,i,h}=\cdots=\beta\_{q,i,h}=0. \\]

You can then compare Korean predictive effects across industries and forecast horizons.

### Critical timing restriction

There are two distinct research questions, and they require different control sets.

|                                  | Post-anchor ARDL            | Total-response ARDL                   |
| -------------------------------- | --------------------------- | ------------------------------------- |
| Dependent variable               | Return after 04:00 or 09:30 | Previous U.S. close to future horizon |
| Korean daily return              | Include                     | Include                               |
| U.S. market return before anchor | Eligible control            | Potential mediator                    |
| ADR spread at anchor             | Eligible if observable      | Not a pre-treatment control           |
| FX movement after Korean close   | Eligible if before anchor   | Potential mediator                    |
| Prior-day macro controls         | Include                     | Include                               |

For the total-response model, including variables observed after the Korean close can remove part of the transmission effect you are trying to measure.

Also, an opening five-minute bar stamped 04:00 is not fully observable until 04:05. Your prediction anchor and feature timestamps must reflect that distinction.

## 6. Proposed control ablation framework

I would use the following nested models for each ADR and horizon.

| Model | Variables added          | Research question                           |
| ----- | ------------------------ | ------------------------------------------- |
| A0    | AR lags + Korean returns | Baseline predictive relationship            |
| A1    | + SPY, KOSPI             | Is the effect independent of broad markets? |
| A2    | + USD/KRW, yields, VIX   | Does it survive macro controls?             |
| A3    | + sector ETF/index       | Is it independent of industry shocks?       |
| A4    | + ADR liquidity features | Does it survive microstructure controls?    |
| A5    | + firm-event indicators  | Is it robust to firm-specific news?         |

These should be cumulative only where the controls satisfy the timing rules. I would also run separate macro-only, sector-only, and liquidity-only ablations to identify which control group changes the Korean coefficient most.

With your current approximately 82-day dataset, the full model would likely be overparameterized. Extend the history to at least 12–24 months, keep the number of regressors small, and use BIC plus walk-forward validation.

## 7. Data acquisition priorities

For your existing Massive → Parquet → DuckDB architecture:

| Data                           | Suggested source                           | Implementation           |
| ------------------------------ | ------------------------------------------ | ------------------------ |
| U.S. ADR prices and volume     | Massive                                    | 5-minute Parquet         |
| SPY, QQQ, XLF, XLU, IYZ, SLX   | Massive                                    | 5-minute Parquet         |
| USD/KRW                        | Massive Forex                              | 5-minute Parquet         |
| ADR bid/ask quotes             | Massive, subject to entitlement            | Tick data                |
| Korean stock prices and KOSPI  | Korean exchange/data provider              | Daily Parquet            |
| Korean yields and macro series | Bank of Korea ECOS                         | Daily/monthly            |
| U.S. yields and macro releases | Treasury, FRED, official release calendars | Timestamped observations |
| Company financial fundamentals | DART, SEC filings                          | Quarterly/event data     |
| Commodities                    | Licensed futures/commodity data provider   | Daily/intraday           |

For fundamental variables, store both the reporting period and public release timestamp. Otherwise, a quarterly financial figure could accidentally enter your backtest before investors actually knew it.

## 8. My recommended implementation order

Phase 1 — Core market controls

Implement all eight pairs with SPY, KOSPI, USD/KRW, and sector ETFs. This provides a consistent baseline across industries.

Phase 2 — Sector-specific controls

Add Korean yields for banks and telecoms, commodity prices for POSCO and KEPCO, and electronics-demand proxies for LG Display.

Phase 3 — Market microstructure

Introduce trade staleness, spreads, relative volume, and opening liquidity. Compare the 04:00 and 09:30 anchors.

Phase 4 — Comparative inference and trading

Estimate Korean-return coefficients across companies, compare out-of-sample improvements, and test whether pricing deviations exceed realistic transaction costs.

For your ultimate arbitrage objective, I would also add a second model alongside ARDL: an ADR fair-value error-correction model using Korean prices, currency conversion, and the verified ADR ratio. ARDL answers whether Korean returns predict U.S. returns; the error-correction model answers whether the ADR is mispriced relative to the Korean underlying and whether that deviation subsequently closes.

That combination is more informative for trading than adding an increasingly large number of macroeconomic variables to ARDL alone.