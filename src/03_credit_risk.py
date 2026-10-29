"""CRISP-DM phases 4-5 (Modeling, Evaluation): delinquency and charge-off forecasts.

For each credit union and loan portfolio, predict next quarter's
  * 60+ day delinquency rate (delinquent balance / portfolio balance)
  * annualized net charge-off rate (quarterly net charge-offs x 4 / balance)
for four portfolios: total loans, credit card, auto, and real estate.

Real estate is spliced across the 2022 call report redesign (see
01_prepare.splice_real_estate). Its model gets two extra features: whether
quarter t is on the new form, and how many of the four lagged quarters come
from the old form, so it can learn the size of the definitional jump instead
of mistaking it for credit deterioration.

Same rolling-origin design as the origination model:
  validation 2022Q3-2024Q2 (model choice), test 2024Q3-2026Q2 (reported).

Baselines: persistence y(t-1) and seasonal persistence y(t-4).
Model: LightGBM regression weighted by portfolio balance.
Metric: balance-weighted MAE in basis points, plus the error of the
balance-weighted portfolio rate summed over all credit unions in scope.
Scope: credit unions with at least $10M in that portfolio at t-1, with four
contiguous prior quarters, and target rows not flagged as anomalies.
"""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MACRO_L1 = ["unrate_l1", "fedfunds_l1", "mortgage30_l1"]
VALID = list(range(30, 38))
TEST = list(range(38, 46))
PORTS = ["total", "card", "auto", "re"]
RE_BREAK_T = 28  # 2022Q1
MIN_BAL = 1e7


def macro_lag1():
    f = pd.read_csv(ROOT / "data" / "raw" / "fred_quarterly.csv")
    f["t"] = (f["quarter"].str[:4].astype(int) - 2015) * 4 + f["quarter"].str[-1].astype(int)
    return f.rename(columns={"unrate": "unrate_l1", "fedfunds": "fedfunds_l1", "mortgage30": "mortgage30_l1"})[["t"] + MACRO_L1]


def frame(df, port, target_kind, add_next=False):
    tgt = f"{target_kind}_rate_{port}"
    cols = ["cu_number", "t", "any_flag", "acct_010", "bal_total", f"bal_{port}",
            f"dq_rate_{port}", f"nco_rate_{port}"]
    d = df[list(dict.fromkeys(cols))].copy()  # dedupe (bal_total appears twice for port=total)
    if add_next:
        last = d[d["t"] == d["t"].max()][["cu_number"]].assign(t=d["t"].max() + 1, any_flag=False)
        d = pd.concat([d, last], ignore_index=True)
    d = d.sort_values(["cu_number", "t"])
    d["qtr"] = (d["t"] % 4) + 1
    d["y"] = d[tgt]
    g = d.groupby("cu_number")
    for k in range(1, 5):
        d[f"dq_l{k}"] = g[f"dq_rate_{port}"].shift(k)
        d[f"nco_l{k}"] = g[f"nco_rate_{port}"].shift(k)
    d["t_l4"] = g["t"].shift(4)
    d["bal_l1"] = g[f"bal_{port}"].shift(1)
    d["log_bal_l1"] = np.log1p(d["bal_l1"])
    d["bal_growth_l1"] = d["bal_l1"] / g[f"bal_{port}"].shift(2) - 1
    d["bal_growth_4q"] = d["bal_l1"] / g[f"bal_{port}"].shift(5) - 1
    d["share_l1"] = d["bal_l1"] / g["bal_total"].shift(1)
    d["log_assets_l1"] = np.log1p(g["acct_010"].shift(1))
    d["flag_l1"] = g["any_flag"].shift(1).astype(float)
    d = d.merge(macro_lag1(), on="t", how="left")
    d["in_scope"] = (d["t_l4"] == d["t"] - 4) & (d["bal_l1"] >= MIN_BAL)
    d["weight"] = d[f"bal_{port}"].fillna(d["bal_l1"])  # balance at t (for scoring) or t-1
    d["y_l1"] = d["dq_l1"] if target_kind == "dq" else d["nco_l1"]
    d["y_l4"] = d["dq_l4"] if target_kind == "dq" else d["nco_l4"]
    d["new_form"] = (d["t"] >= RE_BREAK_T).astype(int)
    d["old_form_lags"] = sum(((d["t"] - k) < RE_BREAK_T).astype(int) for k in range(1, 5)) * d["new_form"]
    return d


FEATS = ["qtr", "dq_l1", "dq_l2", "dq_l3", "dq_l4", "nco_l1", "nco_l2", "nco_l3", "nco_l4",
         "log_bal_l1", "bal_growth_l1", "bal_growth_4q", "share_l1", "log_assets_l1", "flag_l1"] + MACRO_L1


def feats(port):
    return FEATS + (["new_form", "old_form_lags"] if port == "re" else [])


def model():
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                             verbose=-1, random_state=7)


def fit_predict(d, t, port):
    f = feats(port)
    tr = d[(d["t"] < t) & d["in_scope"] & ~d["any_flag"] & d["y"].notna()]
    m = model().fit(tr[f], tr["y"], sample_weight=tr["weight"])
    te = d[(d["t"] == t) & d["in_scope"]].copy()
    te["pred_lgbm"] = m.predict(te[f])
    return te, m


def evaluate(d, origins, port):
    out = []
    for t in origins:
        te, _ = fit_predict(d, t, port)
        te = te[~te["any_flag"] & te["y"].notna() & (te["weight"] > 0)]
        out.append(te[["cu_number", "t", "y", "y_l1", "y_l4", "pred_lgbm", "weight"]])
    return pd.concat(out)


def scores(e):
    res = {}
    for name, col in [("persistence", "y_l1"), ("seasonal_persistence", "y_l4"), ("lgbm", "pred_lgbm")]:
        ok = e[col].notna()
        w = e.loc[ok, "weight"]
        mae = (np.abs(e.loc[ok, "y"] - e.loc[ok, col]) * w).sum() / w.sum()
        # error of the aggregate (balance-weighted) rate, averaged over quarters
        agg = e[ok].assign(a=lambda x: x.y * x.weight, f=lambda x: x[col] * x.weight).groupby("t")[["a", "f", "weight"]].sum()
        agg_err = (np.abs(agg.a - agg.f) / agg.weight).mean()
        res[name] = {"wMAE_bp": mae * 1e4, "aggregate_rate_err_bp": agg_err * 1e4}
    return pd.DataFrame(res).T


def main():
    df = pd.read_parquet(ROOT / "data" / "processed" / "panel.parquet")
    rows, next_fc = [], []
    for kind in ["dq", "nco"]:
        for port in PORTS:
            d = frame(df, port, kind)
            for wname, origins in [("validation", VALID), ("test", TEST)]:
                e = evaluate(d, origins, port)
                sc = scores(e)
                sc["target"], sc["portfolio"], sc["window"] = kind, port, wname
                sc["cu_quarters"] = len(e)
                rows.append(sc.reset_index(names="model"))
            res = pd.concat(rows)
            v = res[(res.target == kind) & (res.portfolio == port) & (res.window == "validation")]
            best = v.sort_values("wMAE_bp")["model"].iloc[0]
            # next quarter (2026Q3) for every in-scope credit union
            dn = frame(df, port, kind, add_next=True)
            t_next = int(df["t"].max()) + 1
            te, _ = fit_predict(dn, t_next, port)
            te["forecast"] = te["pred_lgbm"] if best == "lgbm" else te["y_l1" if best == "persistence" else "y_l4"]
            te["target"], te["portfolio"], te["model_used"] = kind, port, best
            next_fc.append(te[["cu_number", "target", "portfolio", "model_used", "y_l1", "forecast", "bal_l1"]])
            print(f"{kind}/{port}: chosen on validation = {best}")

    res = pd.concat(rows)
    res.to_csv(OUT / "credit_risk_backtest_scores.csv", index=False)
    piv = res.pivot_table(index=["target", "portfolio", "model"], columns="window",
                          values=["wMAE_bp", "aggregate_rate_err_bp"]).round(1)
    print(piv.to_string())
    nf = pd.concat(next_fc).merge(df[df.t == df.t.max()][["cu_number", "cu_name", "state"]], on="cu_number")
    nf.rename(columns={"y_l1": "rate_2026Q2", "forecast": "forecast_2026Q3", "bal_l1": "balance_2026Q2"}).to_csv(
        OUT / "credit_risk_forecast_2026Q3_by_cu.csv", index=False)


if __name__ == "__main__":
    main()
