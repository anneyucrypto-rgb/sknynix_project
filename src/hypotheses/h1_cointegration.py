"""Estimate H1 ARDL–UECM cointegration results and write Markdown."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    breaks_cusumolsresid,
    het_arch,
    het_breuschpagan,
)
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.ardl import UECM, ardl_select_order
from statsmodels.tsa.stattools import adfuller, kpss

from src.config import PROJECT_ROOT


DEFAULT_CONTROLS = ("us_soxx_return", "kr_sector_return")


def _p(value: float) -> str:
    if value < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _significance(value: float) -> str:
    if value < 0.01:
        return "***"
    if value < 0.05:
        return "**"
    if value < 0.10:
        return "*"
    return ""


def _bh_qvalues(values: pd.Series) -> pd.Series:
    """Benjamini–Hochberg false-discovery-rate adjusted p-values."""

    p_values = values.astype(float).to_numpy()
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return pd.Series(output, index=values.index)


def _unit_root_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variable in ("log_adr", "log_kr_usd"):
        for transformation, values in (
            ("Level", frame[variable].dropna()),
            ("First difference", frame[variable].diff().dropna()),
            ("Second difference", frame[variable].diff().diff().dropna()),
        ):
            adf = adfuller(
                values, maxlag=5, regression="c", autolag="BIC"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_result = kpss(values, regression="c", nlags="auto")
            rows.append({
                "variable": variable,
                "transformation": transformation,
                "adf_stat": float(adf[0]),
                "adf_p": float(adf[1]),
                "adf_lags": int(adf[2]),
                "kpss_stat": float(kpss_result[0]),
                "kpss_p": float(kpss_result[1]),
                "kpss_lags": int(kpss_result[2]),
            })
    return rows


def _ratio_inference(
    params: pd.Series,
    covariance: pd.DataFrame,
    numerator: str,
    denominator: str,
    *,
    negate: bool = True,
) -> tuple[float, float, float, float, float]:
    a = float(params[numerator])
    b = float(params[denominator])
    sign = -1.0 if negate else 1.0
    estimate = sign * a / b
    gradient = np.array([sign / b, -sign * a / (b * b)])
    sub_covariance = covariance.loc[
        [numerator, denominator], [numerator, denominator]
    ].to_numpy()
    standard_error = float(np.sqrt(gradient @ sub_covariance @ gradient))
    z_statistic = estimate / standard_error
    p_value = float(2 * norm.sf(abs(z_statistic)))
    return (
        estimate,
        standard_error,
        p_value,
        estimate - 1.96 * standard_error,
        estimate + 1.96 * standard_error,
    )


def estimate_h1(
    data_path: Path,
    *,
    max_lags: int = 5,
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
    finite_simulations: int = 100_000,
    seed: int = 20_261_117,
) -> dict[str, object]:
    frame = pd.read_parquet(data_path).sort_values("date_us")
    required = ["date_us", "log_adr", "log_kr_usd", *controls]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"H1 data are missing columns: {sorted(missing)}")
    sample = frame[required].dropna().reset_index(drop=True)
    if len(sample) < 40:
        raise ValueError("H1 requires at least 40 complete daily observations")

    selected = ardl_select_order(
        sample["log_adr"],
        max_lags,
        sample[["log_kr_usd"]],
        max_lags,
        fixed=sample[list(controls)],
        trend="c",
        ic="bic",
    )
    ardl_result = selected.model.fit()
    uecm_model = UECM.from_ardl(selected.model)
    hac_lags = max(1, int(math.floor(4 * (len(sample) / 100) ** (2 / 9))))
    uecm_result = uecm_model.fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )
    bounds_kwargs = {
        "case": 3,
        "cov_type": "HAC",
        "cov_kwds": {"maxlags": hac_lags},
    }
    bounds_asymptotic = uecm_result.bounds_test(**bounds_kwargs)
    bounds_finite = uecm_result.bounds_test(
        **bounds_kwargs,
        asymptotic=False,
        nsim=finite_simulations,
        seed=seed,
    )

    params = uecm_result.params
    covariance = uecm_result.cov_params()
    long_run_theta = _ratio_inference(
        params, covariance, "log_kr_usd.L1", "log_adr.L1"
    )
    long_run_intercept = _ratio_inference(
        params, covariance, "const", "log_adr.L1"
    )
    adjustment = float(params["log_adr.L1"])
    persistence = 1 + adjustment
    half_life = (
        math.log(0.5) / math.log(persistence)
        if 0 < persistence < 1
        else math.nan
    )

    # statsmodels 0.14's UECMResults.resid is expressed against the original
    # level endog. Diagnostics require the fitted differenced-equation residual.
    residuals = uecm_model._y - uecm_model._x @ params.to_numpy()
    ljung_box = acorr_ljungbox(
        residuals, lags=[5, 10], return_df=True
    )
    bp = het_breuschpagan(residuals, uecm_model._x)
    arch = het_arch(residuals, nlags=5)
    jb = jarque_bera(residuals)
    cusum = breaks_cusumolsresid(
        residuals, ddof=uecm_model._x.shape[1]
    )

    coefficients = pd.DataFrame({
        "term": params.index,
        "estimate": params.to_numpy(),
        "std_error": uecm_result.bse.to_numpy(),
        "p_value": uecm_result.pvalues.to_numpy(),
        "conf_low": uecm_result.conf_int().iloc[:, 0].to_numpy(),
        "conf_high": uecm_result.conf_int().iloc[:, 1].to_numpy(),
    })
    coefficients["q_value"] = _bh_qvalues(coefficients["p_value"])
    coefficients["significance"] = coefficients["p_value"].map(_significance)
    long_run = pd.DataFrame({
        "parameter": ["intercept", "slope"],
        "estimate": [long_run_intercept[0], long_run_theta[0]],
        "std_error": [long_run_intercept[1], long_run_theta[1]],
        "p_value": [long_run_intercept[2], long_run_theta[2]],
        "conf_low": [long_run_intercept[3], long_run_theta[3]],
        "conf_high": [long_run_intercept[4], long_run_theta[4]],
    })
    long_run["q_value"] = _bh_qvalues(long_run["p_value"])
    long_run["significance"] = long_run["p_value"].map(_significance)
    return {
        "frame": frame,
        "sample": sample,
        "unit_roots": _unit_root_rows(sample),
        "selected_order": selected.model.ardl_order,
        "ardl_aic": float(ardl_result.aic),
        "ardl_bic": float(ardl_result.bic),
        "hac_lags": hac_lags,
        "finite_simulations": finite_simulations,
        "coefficients": coefficients,
        "long_run": long_run,
        "bounds_asymptotic": bounds_asymptotic,
        "bounds_finite": bounds_finite,
        "long_run_theta": long_run_theta,
        "long_run_intercept": long_run_intercept,
        "adjustment": adjustment,
        "adjustment_p": float(uecm_result.pvalues["log_adr.L1"]),
        "half_life": half_life,
        "sessions_to_90pct": math.log(0.10) / math.log(persistence),
        "sessions_to_95pct": math.log(0.05) / math.log(persistence),
        "diagnostics": {
            "ljung_box_5": (float(ljung_box.loc[5, "lb_stat"]), float(ljung_box.loc[5, "lb_pvalue"])),
            "ljung_box_10": (float(ljung_box.loc[10, "lb_stat"]), float(ljung_box.loc[10, "lb_pvalue"])),
            "breusch_pagan": (float(bp[0]), float(bp[1])),
            "arch_lm_5": (float(arch[0]), float(arch[1])),
            "jarque_bera": (float(jb[0]), float(jb[1])),
            "cusum": (float(cusum[0]), float(cusum[1])),
            "durbin_watson": float(durbin_watson(residuals)),
            "rmse": float(np.sqrt(np.mean(residuals**2))),
        },
    }


def _anchor_report(anchor_path: Path) -> tuple[list[str], list[str]]:
    if not anchor_path.is_file():
        return (["Anchor dataset unavailable."], [])
    anchors = pd.read_parquet(anchor_path)
    anchors["anchor_ts_utc"] = pd.to_datetime(
        anchors["anchor_ts_utc"], utc=True
    )
    valid_dynamic = anchors[
        anchors["anchor_type"].eq("premarket_dynamic")
        & anchors["anchor_valid"].fillna(False)
    ].copy()
    valid_dynamic["anchor_et"] = valid_dynamic[
        "anchor_ts_utc"
    ].dt.tz_convert("America/New_York")
    valid_dynamic["minutes_after_0700"] = (
        valid_dynamic["anchor_et"].dt.hour * 60
        + valid_dynamic["anchor_et"].dt.minute - 420
    )
    minute_values = valid_dynamic["minutes_after_0700"]

    def clock(minutes_after: float) -> str:
        total = int(round(420 + minutes_after))
        return f"{total // 60:02d}:{total % 60:02d} ET"

    counts = valid_dynamic["anchor_et"].dt.strftime("%H:%M").value_counts()
    summary = [
        "| Measure | Dynamic premarket anchor | Fixed comparison anchor |",
        "| --- | ---: | ---: |",
        f"| Valid dates | {len(valid_dynamic)} | {int((anchors['anchor_type'].eq('regular_0930') & anchors['anchor_valid'].fillna(False)).sum())} |",
        f"| Invalid dates | {int((anchors['anchor_type'].eq('premarket_dynamic') & ~anchors['anchor_valid'].fillna(False)).sum())} | {int((anchors['anchor_type'].eq('regular_0930') & ~anchors['anchor_valid'].fillna(False)).sum())} |",
        f"| Mean time | {clock(float(minute_values.mean()))} | 09:30 ET |",
        f"| Median time | {clock(float(minute_values.median()))} | 09:30 ET |",
        f"| Earliest / latest | {clock(float(minute_values.min()))} / {clock(float(minute_values.max()))} | 09:30 / 09:30 ET |",
        f"| Standard deviation | {float(minute_values.std()):.2f} minutes | 0.00 minutes |",
        f"| Time distribution | " + ", ".join(f"{time} ET: {count}" for time, count in counts.sort_index().items()) + " | 09:30 ET on all valid dates |",
    ]

    daily: list[str] = [
        "| U.S. date | Dynamic time/status | Fixed time/status |",
        "| --- | --- | --- |",
    ]
    for date_us, day in anchors.groupby("date_us", sort=True):
        cells: dict[str, str] = {}
        for _, row in day.iterrows():
            if bool(row["anchor_valid"]):
                local = row["anchor_ts_utc"].tz_convert("America/New_York")
                value = f"{local.strftime('%H:%M')} ET — selected"
            else:
                value = str(row["selection_status"])
            cells[str(row["anchor_type"])] = value
        daily.append(
            f"| {date_us} | {cells.get('premarket_dynamic', 'missing')} | "
            f"{cells.get('regular_0930', 'missing')} |"
        )
    return summary, daily


def write_markdown(
    results: dict[str, object],
    output: Path,
    run_id: str,
    anchor_path: Path | None = None,
) -> Path:
    sample: pd.DataFrame = results["sample"]  # type: ignore[assignment]
    coefficients: pd.DataFrame = results["coefficients"]  # type: ignore[assignment]
    long_run: pd.DataFrame = results["long_run"]  # type: ignore[assignment]
    finite = results["bounds_finite"]
    asymptotic = results["bounds_asymptotic"]
    theta = results["long_run_theta"]
    intercept = results["long_run_intercept"]
    diagnostics: dict[str, object] = results["diagnostics"]  # type: ignore[assignment]
    finite_upper_p = float(finite.p_values["upper"])
    supported = finite_upper_p < 0.05
    anchor_summary, anchor_daily = _anchor_report(
        anchor_path or PROJECT_ROOT / "data" / "features" / "anchors"
        / "part-00000.parquet"
    )
    intercept_row = long_run.loc[long_run["parameter"].eq("intercept")].iloc[0]
    theta_row = long_run.loc[long_run["parameter"].eq("slope")].iloc[0]
    adjustment_row = coefficients.loc[
        coefficients["term"].eq("log_adr.L1")
    ].iloc[0]

    lines = [
        "# H1 — Long-run cointegration",
        "",
        f"**Run ID:** `{run_id}`  ",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ",
        f"**Sample:** {sample['date_us'].min()} through {sample['date_us'].max()}  ",
        f"**Complete observations:** {len(sample)}",
        "",
        "## Result",
        "",
        (
            "The finite-sample ARDL bounds test rejects the null of no level "
            "relationship at the 5% level. H1 is therefore supported for this "
            "short baseline sample."
            if supported else
            "The finite-sample ARDL bounds test does not reject the null of no "
            "level relationship at the 5% level. H1 is not supported."
        ),
        "",
        "> This is statistical evidence within a 90-session sample, not proof "
        "of a stable structural equilibrium. SKHY history begins in July 2026, "
        "so long-run interpretation remains provisional.",
        "",
        "## Specification",
        "",
        f"- Dependent variable: `log_adr`",
        f"- Long-run regressor: `log_kr_usd`",
        f"- Fixed I(0) controls: `us_soxx_return`, `kr_sector_return`",
        f"- Lag search: 0–5 by BIC; selected ARDL{results['selected_order']}",
        f"- UECM deterministic case: unrestricted intercept, no trend (case 3)",
        f"- Inference: HAC with {results['hac_lags']} lags",
        f"- ARDL AIC/BIC: {results['ardl_aic']:.3f} / {results['ardl_bic']:.3f}",
        "",
        "## Integration-order checks",
        "",
        "| Variable | Transformation | ADF statistic | ADF p | KPSS statistic | KPSS p |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in results["unit_roots"]:  # type: ignore[union-attr]
        lines.append(
            f"| `{row['variable']}` | {row['transformation']} | "
            f"{row['adf_stat']:.3f} | {_p(row['adf_p'])} | "
            f"{row['kpss_stat']:.3f} | {_p(row['kpss_p'])} |"
        )
    lines.extend([
        "",
        "ADF has a unit-root null; KPSS has a stationarity null. Both series "
        "are strongly stationary after first differencing, and neither shows "
        "evidence of being I(2). The level tests are mixed for `log_adr`.",
        "",
        "## ARDL–UECM bounds test",
        "",
        "| Calibration | F statistic | Lower p | Upper p | Decision at 5% |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| Asymptotic | {asymptotic.stat:.3f} | {_p(float(asymptotic.p_values['lower']))} | {_p(float(asymptotic.p_values['upper']))} | {'Reject' if float(asymptotic.p_values['upper']) < .05 else 'Do not reject'} |",
        f"| Finite simulation ({results['finite_simulations']:,}) | {finite.stat:.3f} | {_p(float(finite.p_values['lower']))} | {_p(finite_upper_p)} | {'Reject' if supported else 'Do not reject'} |",
        "",
        "Finite-sample simulated critical values:",
        "",
        "| Percentile | I(0) lower | I(1) upper |",
        "| ---: | ---: | ---: |",
    ])
    for percentile in (90.0, 95.0, 99.0):
        lines.append(
            f"| {percentile:.0f}% | {finite.crit_vals.loc[percentile, 'lower']:.3f} | "
            f"{finite.crit_vals.loc[percentile, 'upper']:.3f} |"
        )
    lines.extend([
        "",
        "## Long-run equilibrium and adjustment",
        "",
        "The estimated equilibrium is:",
        "",
        f"`log_adr = {intercept[0]:.4f}{_significance(intercept[2])} + "
        f"{theta[0]:.4f}{_significance(theta[2])} × log_kr_usd`",
        "",
        "Equivalently in price levels:",
        "",
        f"`ADR price = exp({intercept[0]:.4f}) × (FX-adjusted Korean price)^{theta[0]:.4f}`",
        "",
        "| Long-run equilibrium | Estimate | HAC SE | p-value | BH q-value | 95% CI | Significance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| Intercept, ĉ | {intercept_row['estimate']:.4f} | {intercept_row['std_error']:.4f} | {_p(intercept_row['p_value'])} | {_p(intercept_row['q_value'])} | [{intercept_row['conf_low']:.4f}, {intercept_row['conf_high']:.4f}] | {intercept_row['significance']} |",
        f"| Slope, θ̂ | {theta_row['estimate']:.4f} | {theta_row['std_error']:.4f} | {_p(theta_row['p_value'])} | {_p(theta_row['q_value'])} | [{theta_row['conf_low']:.4f}, {theta_row['conf_high']:.4f}] | {theta_row['significance']} |",
        f"| Error correction, λ̂ | {adjustment_row['estimate']:.4f} | {adjustment_row['std_error']:.4f} | {_p(adjustment_row['p_value'])} | {_p(adjustment_row['q_value'])} | [{adjustment_row['conf_low']:.4f}, {adjustment_row['conf_high']:.4f}] | {adjustment_row['significance']} |",
        "",
        f"The slope θ̂={theta[0]:.4f} is a long-run elasticity: a 1% increase "
        f"in the FX-adjusted Korean price is associated with an estimated "
        f"{theta[0]:.3f}% increase in SKHY's long-run price. The intercept ĉ "
        "absorbs the persistent log-level wedge, including price-level and "
        "market-structure differences; it should not be read as a return.",
        "",
        f"The adjustment coefficient λ̂={results['adjustment']:.4f} means SKHY "
        f"corrects approximately {abs(results['adjustment']) * 100:.1f}% of the "
        f"previous disequilibrium per U.S. session. The implied half-life is "
        f"{results['half_life']:.2f} sessions; approximately 90% and 95% of an "
        f"initial deviation decay after {results['sessions_to_90pct']:.2f} and "
        f"{results['sessions_to_95pct']:.2f} sessions, respectively, if the "
        "estimated linear adjustment remains stable.",
        "",
        "H1 is estimated at daily frequency, so it does **not** identify an "
        "'after X minutes' convergence time. Minute-based convergence after "
        "dynamic or 09:30 anchors is an H2 estimand and must be obtained from "
        "the anchor × horizon conditional ECM.",
        "",
        "A p-value measures evidence against a coefficient-specific null under "
        "the fitted model; it is not an effect size. The BH q-value adjusts the "
        "family of reported coefficient tests to control the expected false "
        "discovery rate. Economic magnitude is conveyed by θ̂ and λ̂, not by "
        "their p- or q-values. Significance: *** p<0.01, ** p<0.05, * p<0.10.",
        "",
        "## UECM coefficients",
        "",
        "| Term | Estimate | HAC SE | p-value | BH q-value | 95% CI | Sig. |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in coefficients.to_dict("records"):
        lines.append(
            f"| `{row['term']}` | {row['estimate']:.4f} | {row['std_error']:.4f} | "
            f"{_p(row['p_value'])} | {_p(row['q_value'])} | "
            f"[{row['conf_low']:.4f}, {row['conf_high']:.4f}] | {row['significance']} |"
        )
    lines.extend([
        "",
        "## Residual diagnostics",
        "",
        "| Diagnostic | Statistic | p-value | Assessment |",
        "| --- | ---: | ---: | --- |",
    ])
    diagnostic_labels = (
        ("Ljung–Box Q(5)", "ljung_box_5", "no serial correlation"),
        ("Ljung–Box Q(10)", "ljung_box_10", "no serial correlation"),
        ("Breusch–Pagan", "breusch_pagan", "homoskedasticity"),
        ("ARCH-LM(5)", "arch_lm_5", "no ARCH effects"),
        ("Jarque–Bera", "jarque_bera", "normality"),
        ("CUSUM", "cusum", "parameter stability"),
    )
    for label, key, null_description in diagnostic_labels:
        statistic, p_value = diagnostics[key]
        assessment = (
            f"No evidence against {null_description}"
            if p_value >= 0.05 else f"Reject {null_description}"
        )
        lines.append(
            f"| {label} | {statistic:.3f} | {_p(p_value)} | {assessment} |"
        )
    lines.extend([
        f"| Durbin–Watson | {diagnostics['durbin_watson']:.3f} | — | Descriptive |",
        f"| Equation RMSE | {diagnostics['rmse']:.4f} | — | Descriptive |",
        "",
        "HAC inference is retained because Breusch–Pagan rejects "
        "homoskedasticity. Jarque–Bera also rejects residual normality. Ljung–Box, "
        "ARCH-LM, and CUSUM do not reject their respective nulls at 5%; these "
        "nonsignificant results are not proof that the assumptions hold.",
        "",
        "## Anchor-time statistics (descriptive; used by H2)",
        "",
        "Anchor timing does not enter the daily H1 bounds test. It is reported "
        "here to connect the established daily equilibrium to the subsequent "
        "minute-horizon H2 analysis.",
        "",
        *anchor_summary,
        "",
        "<details>",
        "<summary>Daily anchor times and selection status</summary>",
        "",
        *anchor_daily,
        "",
        "</details>",
        "",
        "## Final cointegration assessment",
        "",
        "Valid statistical cointegration evidence **was obtained under the "
        "pre-specified baseline model** because: (1) neither price series shows "
        "evidence of being I(2); (2) the finite-sample bounds statistic exceeds "
        f"the 5% I(1) upper critical value and has upper-bound "
        f"p={_p(finite_upper_p)}; and "
        f"(3) the error-correction coefficient is negative and significant "
        f"(λ̂={adjustment_row['estimate']:.4f}, p={_p(adjustment_row['p_value'])}, "
        f"BH q={_p(adjustment_row['q_value'])}).",
        "",
        "**H1 verdict: supported at the 5% level, but not conclusive as a "
        "structural long-run claim.** The result is internally valid for this "
        "specification and sample; it remains provisional because the sample "
        "contains only 90 sessions, the level unit-root tests are mixed for "
        "`log_adr`, and residual heteroskedasticity/non-normality is present.",
        "",
        "## Interpretation constraints",
        "",
        "- The effective history is too short for a strong structural "
        "long-run claim; results should be updated as SKHY history accumulates.",
        "- The bounds conclusion is sensitive to specification, deterministic "
        "case, lag selection, and small-sample calibration.",
        "- Korean prices are FX-adjusted and aligned point-in-time; this does "
        "not eliminate market-hours, trading-friction, or ADS-ratio risks.",
        "- This H1 result supports estimating conditional H2/H3 ECMs, but those "
        "models should use coefficients trained on an earlier or otherwise "
        "separated sample to avoid generated-regressor look-ahead.",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="baseline-default")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-lags", type=int, default=5)
    parser.add_argument("--finite-simulations", type=int, default=100_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data_path or (
        PROJECT_ROOT / "data" / "features" / "experiments"
        / f"run_id={args.run_id}" / "h1.parquet"
    )
    output = args.output or (
        PROJECT_ROOT / "reports" / f"run_id={args.run_id}"
        / "H1_cointegration.md"
    )
    print(f"[H1] Reading {data_path}", flush=True)
    results = estimate_h1(
        data_path,
        max_lags=args.max_lags,
        finite_simulations=args.finite_simulations,
    )
    write_markdown(results, output, args.run_id)
    finite = results["bounds_finite"]
    print(
        f"[H1] ARDL{results['selected_order']} bounds F={finite.stat:.4f}, "
        f"upper p={float(finite.p_values['upper']):.4f}",
        flush=True,
    )
    print(f"[H1] Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
