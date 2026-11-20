"""Statistical routines for the Experimentation & Causal Inference Workbench."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportions_ztest


def sample_size(effect: float, alpha: float = .05, power: float = .8, ratio: float = 1.0) -> int:
    """Approximate per-arm size for a two-sided independent means test."""
    if effect <= 0 or not 0 < alpha < 1 or not 0 < power < 1 or ratio <= 0:
        raise ValueError("Effect and allocation ratio must be positive; alpha and power must be between zero and one.")
    return int(math.ceil(TTestIndPower().solve_power(effect_size=effect, alpha=alpha, power=power, ratio=ratio)))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    return multipletests(p_values, method="fdr_bh")[1].tolist()


def analyze_experiment(df: pd.DataFrame, group: str, outcome: str, kind: str, control: str, treatment: str,
                       alpha: float = .05) -> dict[str, Any]:
    data = df[[group, outcome]].dropna()
    a = data.loc[data[group].astype(str) == control, outcome].astype(float).to_numpy()
    b = data.loc[data[group].astype(str) == treatment, outcome].astype(float).to_numpy()
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each selected arm must contain at least two non-missing observations.")
    if kind == "Binary conversion":
        if not set(np.unique(np.r_[a, b])).issubset({0, 1}):
            raise ValueError("Binary outcomes must contain only 0/1 values.")
        counts = np.array([b.sum(), a.sum()])
        nobs = np.array([len(b), len(a)])
        _, p = proportions_ztest(counts, nobs)
        rate_a, rate_b = a.mean(), b.mean()
        diff = rate_b - rate_a
        se = math.sqrt(rate_a * (1-rate_a)/len(a) + rate_b * (1-rate_b)/len(b))
        ci = (diff - 1.96*se, diff + 1.96*se)
        effect = diff / math.sqrt(max((rate_a*(1-rate_a)+rate_b*(1-rate_b))/2, 1e-12))
        test = "Two-proportion z-test"
    else:
        result = stats.ttest_ind(b, a, equal_var=False, alternative="two-sided")
        p = float(result.pvalue)
        diff = float(b.mean() - a.mean())
        se = math.sqrt(b.var(ddof=1)/len(b) + a.var(ddof=1)/len(a))
        df_w = (b.var(ddof=1)/len(b)+a.var(ddof=1)/len(a))**2 / ((b.var(ddof=1)/len(b))**2/(len(b)-1)+(a.var(ddof=1)/len(a))**2/(len(a)-1))
        crit = stats.t.ppf(1-alpha/2, df_w)
        ci = (diff-crit*se, diff+crit*se)
        pooled = math.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
        effect = diff / pooled if pooled else 0.0
        test = "Welch's t-test"
    return {"n_a": len(a), "n_b": len(b), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "effect": float(diff), "ci_low": float(ci[0]), "ci_high": float(ci[1]), "p_value": float(p),
            "significant": bool(p < alpha), "test": test, "standardized_effect": float(effect)}


def analyze_categorical(df: pd.DataFrame, group: str, outcome: str, control: str,
                        treatment: str, alpha: float = .05) -> dict[str, Any]:
    """Compare a categorical outcome distribution across two experiment arms."""
    data = df[[group, outcome]].dropna().copy()
    data[group] = data[group].astype(str)
    data = data[data[group].isin([control, treatment])]
    data[outcome] = data[outcome].astype(str)
    table = pd.crosstab(data[group], data[outcome]).reindex(index=[control, treatment], fill_value=0)
    table = table.loc[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        raise ValueError("The selected outcome needs at least two observed categories.")
    if (table.sum(axis=1) == 0).any():
        raise ValueError("Both selected arms need at least one non-missing outcome.")
    chi2, p_value, dof, expected = stats.chi2_contingency(table.to_numpy(), correction=False)
    n = int(table.to_numpy().sum())
    denominator = n * min(table.shape[0]-1, table.shape[1]-1)
    cramer_v = math.sqrt(float(chi2) / denominator) if denominator > 0 else 0.0
    return {"n_control": int(table.loc[control].sum()), "n_treatment": int(table.loc[treatment].sum()),
            "statistic": float(chi2), "degrees_freedom": int(dof), "p_value": float(p_value),
            "significant": bool(p_value < alpha), "cramers_v": cramer_v,
            "expected_min": float(np.min(expected)), "contingency": table}


def sequential_analysis(df: pd.DataFrame, group: str, outcome: str, kind: str,
                        control: str, treatment: str, looks: int = 5,
                        alpha: float = .05) -> pd.DataFrame:
    """Evaluate a fixed schedule of cumulative looks with Bonferroni FWER control.

    Rows must already be sorted in arrival order. Each nominal test is evaluated
    at alpha / looks, so the probability of one or more false rejections across
    the pre-specified looks is at most alpha, under the validity assumptions of
    the underlying per-look tests.
    """
    if looks < 2:
        raise ValueError("Plan at least two looks.")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between zero and one.")
    data = df[[group, outcome]].dropna().reset_index(drop=True)
    if len(data) < 4:
        raise ValueError("Need at least four complete rows for sequential analysis.")
    rows: list[dict[str, Any]] = []
    for look in range(1, looks + 1):
        stop = max(1, int(np.floor(len(data) * look / looks)))
        snapshot = data.iloc[:stop]
        selected = snapshot[snapshot[group].astype(str).isin([control, treatment])]
        n_control = int((selected[group].astype(str) == control).sum())
        n_treatment = int((selected[group].astype(str) == treatment).sum())
        row: dict[str, Any] = {"Look": look, "Rows observed": stop,
                               "Control n": n_control, "Treatment n": n_treatment,
                               "Raw p-value": np.nan, "Adjusted p-value": np.nan,
                               "Effect": np.nan, "Decision": "Insufficient data"}
        if n_control >= 2 and n_treatment >= 2:
            try:
                result = analyze_experiment(selected, group, outcome, kind, control, treatment, alpha)
                p_adjusted = min(1.0, result["p_value"] * looks)
                row.update({"Raw p-value": result["p_value"],
                            "Adjusted p-value": p_adjusted,
                            "Effect": result["effect"],
                            "Decision": "Crossed boundary" if p_adjusted <= alpha else "Continue"})
            except ValueError:
                row["Decision"] = "Check outcome values"
        rows.append(row)
    return pd.DataFrame(rows)


def beta_binary_posterior(a: np.ndarray, b: np.ndarray, draws: int = 100_000) -> dict[str, float]:
    """Conjugate Beta(1,1) conversion model (exact sampling, no MCMC required)."""
    ra = np.random.default_rng(201)
    pa = ra.beta(float(a.sum())+1, float(len(a)-a.sum())+1, draws)
    pb = ra.beta(float(b.sum())+1, float(len(b)-b.sum())+1, draws)
    delta = pb-pa
    return {"prob_b_better": float(np.mean(delta > 0)), "expected_lift": float(np.mean(delta)),
            "ci_low": float(np.quantile(delta, .025)), "ci_high": float(np.quantile(delta, .975))}


def difference_in_differences(df: pd.DataFrame, outcome: str, group: str, period: str,
                              treated: str, post: str) -> dict[str, float]:
    means = df.groupby([group, period])[outcome].mean().unstack()
    if treated not in means.index or post not in means.columns or len(means.columns) < 2:
        raise ValueError("Choose a dataset with treated/control groups and observations before and after treatment.")
    control = next((x for x in means.index if x != treated), None)
    post_index = list(means.columns).index(post)
    pre = list(means.columns[:post_index])
    post_periods = list(means.columns[post_index:])
    if control is None or not pre or not post_periods:
        raise ValueError("Could not identify comparison group and pre-period.")
    treated_pre = float(means.loc[treated, pre].mean())
    control_pre = float(means.loc[control, pre].mean())
    treated_post = float(means.loc[treated, post_periods].mean())
    control_post = float(means.loc[control, post_periods].mean())
    effect = (treated_post-treated_pre)-(control_post-control_pre)
    return {"effect": float(effect), "treated_pre": treated_pre,
            "treated_post": treated_post, "control_pre": control_pre,
            "control_post": control_post}


def synthetic_control(df: pd.DataFrame, unit: str, time: str, outcome: str, treated: str, intervention: Any) -> pd.DataFrame:
    wide = df.pivot_table(index=time, columns=unit, values=outcome, aggfunc="mean").sort_index()
    donors = [c for c in wide.columns if c != treated]
    pre = wide.index < intervention
    if treated not in wide or not pre.any() or not (~pre).any() or not donors:
        raise ValueError("Need a treated unit, donor units, and observations before and after intervention.")
    x = wide.loc[pre, donors].to_numpy()
    y = wide.loc[pre, treated].to_numpy()
    from scipy.optimize import minimize
    n = len(donors)
    fit = minimize(lambda w: float(np.mean((x @ w-y)**2)), np.full(n, 1/n), method="SLSQP",
                   bounds=[(0, 1)]*n, constraints=[{"type": "eq", "fun": lambda w: w.sum()-1}],
                   options={"maxiter": 1000, "ftol": 1e-12})
    weights = np.clip(fit.x, 0, None); weights /= weights.sum()
    synth = wide[donors].to_numpy() @ weights
    return pd.DataFrame({"time": wide.index, "Observed treated": wide[treated].to_numpy(),
                         "Synthetic control": synth, "Estimated effect": wide[treated].to_numpy()-synth})
