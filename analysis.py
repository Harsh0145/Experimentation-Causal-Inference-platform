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
    if treated not in means.index or post not in means.columns or len(means.columns) != 2:
        raise ValueError("Choose a two-period dataset with treated/control groups and pre/post periods.")
    control = next((x for x in means.index if x != treated), None)
    pre = next((x for x in means.columns if x != post), None)
    if control is None or pre is None:
        raise ValueError("Could not identify comparison group and pre-period.")
    effect = (means.loc[treated, post]-means.loc[treated, pre])-(means.loc[control, post]-means.loc[control, pre])
    return {"effect": float(effect), "treated_pre": float(means.loc[treated, pre]),
            "treated_post": float(means.loc[treated, post]), "control_pre": float(means.loc[control, pre]),
            "control_post": float(means.loc[control, post])}


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
