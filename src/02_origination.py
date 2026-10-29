"""CRISP-DM phases 4-5 (Modeling, Evaluation): new loan origination forecasts.

Target: quarterly dollar amount of loans granted (NCUA account 031B,
de-cumulated from year-to-date). Forecast at two levels:
  * industry total across all federally insured credit unions
  * each individual credit union

Evaluation design (rolling origin, one quarter ahead, refit at every origin):
  validation window  2022Q3-2024Q2  -> used to choose models
  test window        2024Q3-2026Q2  -> reported, not used for choices

Industry models
  seasonal_naive        y(t) = y(t-4)
  seasonal_naive_drift  y(t-4) x (last 4 quarters / prior 4 quarters)
  sarimax_macro         log y: AR(1), seasonal difference + MA(1) at lag 4,
                        exogenous macro drivers from quarter t-1
  panel_tweedie, panel_growth  credit-union models below, summed

Credit-union models (rows with 8 contiguous prior quarters)
  seasonal_naive  g(t) = g(t-4)
  panel_tweedie   LightGBM, Tweedie loss, predicts the dollar level
  panel_growth    LightGBM, L1 (median) loss, predicts log growth versus the
                  same quarter last year; forecast = g(t-4) x exp(prediction).
                  Matches the WAPE metric (absolute error) and lets the model
                  learn changes rather than re-learn each credit union's scale.
"""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
MACRO = ["unrate", "fedfunds", "mortgage30"]
MACRO_L1 = [f"{c}_l1" for c in MACRO]
VALID = list(range(30, 38))  # 2022Q3..2024Q2
TEST = list(range(38, 46))   # 2024Q3..2026Q2
EPS = 0.05                   # $50K offset so log growth is defined for zero quarters


def macro_lag1():
    """FRED values from quarter t-1, keyed by the panel time index t."""
    f = pd.read_csv(ROOT / "data" / "raw" / "fred_quarterly.csv")
    t_of_quarter = (f["quarter"].str[:4].astype(int) - 2015) * 4 + f["quarter"].str[-1].astype(int) - 1
    f["t"] = t_of_quarter + 1
    return f.rename(columns=dict(zip(MACRO, MACRO_L1)))[["t"] + MACRO_L1]


def industry_series(df):
    s = (df.groupby(["t", "period"], as_index=False)
         .agg(granted=("q_granted_amt", "sum"), n_cu=("cu_number", "nunique"))
         .sort_values("t"))
    s = s[s["t"] >= 2]  # 2015Q2 (t=1) is the first file, so it has no quarterly flow
    return s.merge(macro_lag1(), on="t", how="left").reset_index(drop=True)


BASE_FEATS = ["qtr", "g_l1", "g_l2", "g_l3", "g_l4", "log_assets_l1", "log_loans_l1",
              "loan_growth_l1", "dq_l1"] + MACRO_L1
GROWTH_FEATS = BASE_FEATS + ["g_l5", "drift", "ratio_l1_l5"]


def panel_frame(df, add_next=False):
    d = df[["cu_number", "t", "q_granted_amt", "acct_010", "bal_total", "dq_rate_total", "any_flag"]].copy()
    if add_next:  # one empty row per active credit union for the next quarter
        last = d[d["t"] == d["t"].max()][["cu_number"]].assign(t=d["t"].max() + 1, any_flag=False)
        d = pd.concat([d, last], ignore_index=True)
    d = d.sort_values(["cu_number", "t"])
    d["qtr"] = (d["t"] % 4) + 1
    d["g"] = d["q_granted_amt"].clip(lower=0) / 1e6  # $ millions
    g = d.groupby("cu_number")
    for k in range(1, 9):
        d[f"g_l{k}"] = g["g"].shift(k)
    d["t_l8"] = g["t"].shift(8)
    d["log_assets_l1"] = np.log1p(g["acct_010"].shift(1))
    d["log_loans_l1"] = np.log1p(g["bal_total"].shift(1))
    d["loan_growth_l1"] = g["bal_total"].shift(1) / g["bal_total"].shift(2) - 1
    d["dq_l1"] = g["dq_rate_total"].shift(1)
    last4 = d[[f"g_l{k}" for k in range(1, 5)]].sum(axis=1)
    prev4 = d[[f"g_l{k}" for k in range(5, 9)]].sum(axis=1)
    d["drift"] = (last4 + EPS) / (prev4 + EPS)
    d["ratio_l1_l5"] = (d["g_l1"] + EPS) / (d["g_l5"] + EPS)
    d["log_growth"] = np.log((d["g"] + EPS) / (d["g_l4"] + EPS))
    d = d.merge(macro_lag1(), on="t", how="left")
    d["in_scope"] = d["t_l8"] == d["t"] - 8  # 8 contiguous prior quarters
    return d


def tweedie_model():
    return lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, n_estimators=600,
                             learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=7)


def growth_model():
    return lgb.LGBMRegressor(objective="l1", n_estimators=600, learning_rate=0.05, num_leaves=31,
                             min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             verbose=-1, random_state=7)


def panel_predict(p, t):
    """Fit both credit-union models on rows before t; predict every credit union at t ($M).

    Credit unions without 8 contiguous prior quarters fall back to seasonal
    naive (or zero if they have no same-quarter-last-year value).
    """
    train = p[(p["t"] < t) & p["in_scope"] & ~p["any_flag"] & p["g"].notna()]
    tw = tweedie_model().fit(train[BASE_FEATS], train["g"])
    gr = growth_model().fit(train[GROWTH_FEATS], train["log_growth"])
    cur = p[p["t"] == t].copy()
    fallback = cur["g_l4"].fillna(0)
    ok = cur["in_scope"]
    cur["pred_snaive"] = fallback
    cur["pred_tweedie"] = fallback
    cur["pred_growth"] = fallback
    cur.loc[ok, "pred_tweedie"] = tw.predict(cur.loc[ok, BASE_FEATS]).clip(min=0)
    growth = gr.predict(cur.loc[ok, GROWTH_FEATS])
    cur.loc[ok, "pred_growth"] = ((cur.loc[ok, "g_l4"] + EPS) * np.exp(growth) - EPS).clip(lower=0)
    return cur


def run_window(s, p, origins):
    rows, cu_rows = [], []
    for t in origins:
        hist = s[s["t"] < t]
        actual = s.loc[s["t"] == t, "granted"].item()
        y4 = s.loc[s["t"] == t - 4, "granted"].item()
        fc = {"seasonal_naive": y4,
              "seasonal_naive_drift": y4 * hist["granted"].iloc[-4:].sum() / hist["granted"].iloc[-8:-4].sum()}
        m = SARIMAX(np.log(hist["granted"].values), exog=hist[MACRO_L1].values, order=(1, 0, 0),
                    seasonal_order=(0, 1, 1, 4), trend="c",
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc["sarimax_macro"] = float(np.exp(m.forecast(1, exog=s.loc[s["t"] == t, MACRO_L1].values)[0]))
        cur = panel_predict(p, t)
        fc["panel_tweedie"] = cur["pred_tweedie"].sum() * 1e6
        fc["panel_growth"] = cur["pred_growth"].sum() * 1e6
        cu_rows.append(cur.loc[cur["in_scope"] & ~cur["any_flag"] & cur["g"].notna(),
                               ["cu_number", "t", "g", "pred_snaive", "pred_tweedie", "pred_growth"]])
        period = s.loc[s["t"] == t, "period"].item()
        for k, v in fc.items():
            rows.append({"period": period, "t": t, "model": k, "forecast": v, "actual": actual})
    bt = pd.DataFrame(rows)
    bt["ape"] = (bt["forecast"] - bt["actual"]).abs() / bt["actual"]
    return bt, pd.concat(cu_rows)


def wape(a, f):
    return np.abs(a - f).sum() / np.abs(a).sum()


def main():
    df = pd.read_parquet(ROOT / "data" / "processed" / "panel.parquet")
    s = industry_series(df)
    p = panel_frame(df)
    s.to_csv(OUT / "origination_industry_series.csv", index=False)

    results = {}
    for name, origins in [("validation", VALID), ("test", TEST)]:
        bt, cu = run_window(s, p, origins)
        bt["window"] = name
        results[name] = (bt, cu)
    bt_all = pd.concat([results[w][0] for w in results])
    bt_all.to_csv(OUT / "origination_backtest.csv", index=False)

    summ = (bt_all.groupby(["window", "model"])["ape"].agg(MAPE="mean", worst_APE="max")
            .unstack("window").swaplevel(axis=1).sort_index(axis=1))
    print("Industry quarterly loans granted: one-quarter-ahead rolling backtest, APE in %")
    print((summ * 100).round(2).to_string())
    summ.to_csv(OUT / "origination_model_summary.csv")
    best = bt_all[bt_all.window == "validation"].groupby("model")["ape"].mean().idxmin()
    print(f"industry model chosen on validation window: {best}")

    cu_tbl = []
    for w, (_, cu) in results.items():
        row = {"window": w, "cu_quarters": len(cu)}
        for m in ["snaive", "tweedie", "growth"]:
            row[f"WAPE_{m}"] = wape(cu["g"], cu[f"pred_{m}"])
        cu_tbl.append(row)
    cu_tbl = pd.DataFrame(cu_tbl)
    cu_tbl.to_csv(OUT / "origination_cu_level_wape.csv", index=False)
    print("Credit-union level WAPE (unflagged rows with 8 contiguous prior quarters):")
    print(cu_tbl.round(4).to_string(index=False))
    v = cu_tbl[cu_tbl.window == "validation"].iloc[0]
    best_cu = min(["snaive", "tweedie", "growth"], key=lambda m: v[f"WAPE_{m}"])
    print(f"credit-union model chosen on validation window: {best_cu}")

    # next quarter (2026Q3): industry total with the chosen industry model
    assert best == "sarimax_macro", best
    m = SARIMAX(np.log(s["granted"].values), exog=s[MACRO_L1].values, order=(1, 0, 0),
                seasonal_order=(0, 1, 1, 4), trend="c",
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    x_next = macro_lag1().query("t == @s.t.max() + 1")[MACRO_L1].values
    f = m.get_forecast(1, exog=x_next)
    lo, hi = np.exp(f.conf_int(alpha=0.2)[0])
    point = float(np.exp(f.predicted_mean[0]))
    last_year = s["granted"].iloc[-4]
    pd.DataFrame([{"period": "2026Q3", "model": best, "forecast_usd": point, "pi80_low_usd": lo,
                   "pi80_high_usd": hi, "same_quarter_last_year_usd": last_year}]).to_csv(
        OUT / "origination_forecast_2026Q3.csv", index=False)
    print(f"2026Q3 industry forecast ({best}): ${point/1e9:.1f}B (80% PI ${lo/1e9:.1f}B-${hi/1e9:.1f}B); "
          f"2025Q3 actual ${last_year/1e9:.1f}B")

    # next quarter (2026Q3): each credit union with the chosen credit-union model
    pn = panel_frame(df, add_next=True)
    cur = panel_predict(pn, int(df["t"].max()) + 1)
    cur = cur.merge(df[df.t == df.t.max()][["cu_number", "cu_name", "state"]], on="cu_number")
    out = cur[["cu_number", "cu_name", "state", "in_scope", "g_l4", f"pred_{best_cu}"]].rename(columns={
        "in_scope": "modeled", "g_l4": "granted_2025Q3_musd", f"pred_{best_cu}": "forecast_2026Q3_musd"})
    out["model"] = np.where(out["modeled"], f"panel_{best_cu}", "seasonal_naive_fallback")
    out.sort_values("forecast_2026Q3_musd", ascending=False).to_csv(
        OUT / "origination_forecast_2026Q3_by_cu.csv", index=False)
    print(f"credit-union forecasts written: {len(out):,} ({out.modeled.sum():,} modeled)")


if __name__ == "__main__":
    main()
