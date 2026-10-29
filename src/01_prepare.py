"""CRISP-DM phase 3 (Data Preparation).

Builds a clean credit-union x quarter panel from the NCUA 5300 Call Report
extract (2015 Q2 - 2026 Q2) and FRED quarterly macro series.

Steps
1. Cast account fields to numbers and build quarter keys.
2. De-cumulate year-to-date (YTD) flow accounts into quarterly flows.
3. Build portfolio balances, 60+ day delinquency amounts and net charge-offs.
4. Flag erroneous or outlier records (logic checks, robust z-scores, suspected
   mergers) and write an anomaly log.
5. Join FRED macro drivers.

Output: data/processed/panel.parquet, data/processed/anomaly_log.csv
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Year-to-date flow accounts (NCUA reports these cumulatively within a calendar year)
YTD = {
    "acct_031a": "granted_n",       # number of loans granted YTD
    "acct_031b": "granted_amt",     # amount of loans granted YTD
    "acct_550": "co_total",         # total charge-offs YTD
    "acct_551": "rec_total",        # total recoveries YTD
    "acct_680": "co_card",
    "acct_681": "rec_card",
    "acct_550c1": "co_newveh",
    "acct_550c2": "co_usedveh",
    "acct_551c1": "rec_newveh",
    "acct_551c2": "rec_usedveh",
    "acct_co_re": "co_re",           # real estate charge-offs (spliced across the 2022 redesign)
    "acct_rec_re": "rec_re",
}
RE_BREAK_T = 28  # 2022Q1: first quarter on the redesigned 5300 form


def splice_real_estate(df: pd.DataFrame) -> pd.DataFrame:
    """Real estate accounts were redefined in the 2022 call report redesign.

    Through 2021Q4: all 1st mortgage + other real estate loans (703 + 386);
      60+ day delinquency = the 2-6, 6-12 and 12+ month buckets for fixed and
      adjustable 1st mortgage (752-754, 772-774) and other real estate
      (756-758, 776-778); charge-offs 548 + 549, recoveries 607 + 608.
    From 2022Q1: 1-4 family residential, 1st lien + junior lien (703A + 386A);
      60+ day ("reportable") delinquency DL0062 + DL0069; charge-offs
      CH0017 + CH0019, recoveries CH0018 + CH0020.
    Year-to-date charge-offs reset every January, and the break falls on a
    year boundary, so de-cumulation never mixes the two definitions.
    """
    old = df["t"] < RE_BREAK_T
    s = lambda cols: df[cols].sum(axis=1, min_count=len(cols))
    df["acct_co_re"] = np.where(old, s(["acct_548", "acct_549"]), s(["acct_ch0017", "acct_ch0019"]))
    df["acct_rec_re"] = np.where(old, s(["acct_607", "acct_608"]), s(["acct_ch0018", "acct_ch0020"]))
    old_dq = ["acct_752", "acct_753", "acct_754", "acct_772", "acct_773", "acct_774",
              "acct_756", "acct_757", "acct_758", "acct_776", "acct_777", "acct_778"]
    df["dq_re"] = np.where(old, s(old_dq), s(["acct_dl0062", "acct_dl0069"]))
    df["bal_re"] = np.where(old, s(["acct_703", "acct_386"]), s(["acct_703a", "acct_386a"]))
    df["re_new_form"] = (~old).astype(int)
    return df


def load_raw() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(
        f"create table r as select * from read_csv_auto('{RAW / 'ncua_panel.csv.gz'}', all_varchar=true)"
    )
    cols = con.execute("select * from r limit 0").df().columns
    sel = ", ".join(
        f"try_cast({c} as double) as {c}" if c.startswith("acct_") else c for c in cols
    )
    df = con.execute(f"select {sel} from r").df()
    # second extract: real estate delinquency buckets and charge-offs (old and 2022+ form)
    con.execute(
        f"create table re as select * from read_csv_auto('{RAW / 'ncua_re_panel.csv.gz'}', all_varchar=true)"
    )
    re_cols = [c for c in con.execute("select * from re limit 0").df().columns if c.startswith("acct_")]
    re_cols = [c for c in re_cols if c not in df.columns]
    re = con.execute("select quarter, cu_number, " + ", ".join(f"try_cast({c} as double) as {c}" for c in re_cols)
                     + " from re").df()
    df = df.merge(re, on=["quarter", "cu_number"], how="left")
    df["cu_number"] = df["cu_number"].astype(int)
    df["year"] = df["quarter"].str[:4].astype(int)
    df["qtr"] = (df["quarter"].str[5:7].astype(int) // 3).astype(int)
    df["period"] = df["year"].astype(str) + "Q" + df["qtr"].astype(str)
    df["t"] = (df["year"] - 2015) * 4 + df["qtr"] - 1  # integer time index
    return df.sort_values(["cu_number", "t"]).reset_index(drop=True)


def decumulate(df: pd.DataFrame) -> pd.DataFrame:
    """Convert YTD accounts to quarterly flows.

    Q1 flow = Q1 YTD. Qn flow = YTD(n) - YTD(n-1) for the same CU in the same
    year. If the prior quarter is missing (first quarter on file, or a gap),
    the flow is left missing rather than guessed.
    """
    g = df.groupby("cu_number")
    prev_t = g["t"].shift(1)
    contiguous = (prev_t == df["t"] - 1) & (g["year"].shift(1) == df["year"])
    for src, name in YTD.items():
        prev = g[src].shift(1)
        q_flow = np.where(df["qtr"] == 1, df[src], np.where(contiguous, df[src] - prev, np.nan))
        df[f"q_{name}"] = q_flow
    return df


def build_measures(df: pd.DataFrame) -> pd.DataFrame:
    df["bal_total"] = df["acct_025b"]
    df["bal_card"] = df["acct_396"]
    df["bal_auto"] = df["acct_385"] + df["acct_370"]
    df["bal_unsec_other"] = df["acct_397"]
    # 1-4 family 1st lien / junior lien balances exist on a consistent basis
    # from 2017 Q3 (accounts 703A/386A). The older 703/386 series stop in 2022.
    df["bal_mortgage"] = np.where(df["t"] >= 9, df["acct_703a"], np.nan)
    df["bal_heloc_junior"] = np.where(df["t"] >= 9, df["acct_386a"], np.nan)

    # 60+ day ("reportable") delinquency amounts
    df["dq_total"] = df["acct_041b"]
    df["dq_card"] = df["acct_045b"]
    df["dq_auto"] = df["acct_041c1"] + df["acct_041c2"]

    # quarterly net charge-offs
    df["q_nco_total"] = df["q_co_total"] - df["q_rec_total"]
    df["q_nco_card"] = df["q_co_card"] - df["q_rec_card"]
    df["q_nco_auto"] = (df["q_co_newveh"] + df["q_co_usedveh"]) - (
        df["q_rec_newveh"] + df["q_rec_usedveh"]
    )
    df["q_nco_re"] = df["q_co_re"] - df["q_rec_re"]

    def rate(num, den):
        return np.where(df[den] > 0, df[num] / df[den], np.nan)

    for p in ["total", "card", "auto", "re"]:
        df[f"dq_rate_{p}"] = rate(f"dq_{p}", f"bal_{p}")
        # annualized net charge-off rate on the quarter-end balance
        df[f"nco_rate_{p}"] = 4 * rate(f"q_nco_{p}", f"bal_{p}")
    return df


def flag_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logs = []

    def log(mask, rule, field):
        sub = df.loc[mask, ["cu_number", "cu_name", "period", field]].copy()
        sub["rule"] = rule
        sub = sub.rename(columns={field: "value"})
        sub["field"] = field
        logs.append(sub)
        return mask

    flags = pd.DataFrame(index=df.index)

    # 1. Logic checks
    for p in ["total", "card", "auto", "re"]:
        m = df[f"dq_{p}"] > df[f"bal_{p}"] + 1
        flags[f"f_dq_gt_bal_{p}"] = log(m, "delinquent amount exceeds balance", f"dq_{p}")
    for name in ["granted_amt", "co_total", "rec_total"]:
        m = df[f"q_{name}"] < 0
        flags[f"f_neg_{name}"] = log(m, "YTD value fell vs prior quarter (negative quarterly flow)", f"q_{name}")
    m = df["q_co_total"] > df["bal_total"] + 1
    flags["f_co_gt_bal"] = log(m, "quarterly charge-offs exceed loan balance", "q_co_total")

    # 2. Suspected mergers: CU exits after this quarter, and survivors with an
    #    inorganic jump in assets (>25% QoQ) in the same quarter.
    g = df.groupby("cu_number")
    last_t = g["t"].transform("max")
    df["exited_next_q"] = (df["t"] == last_t) & (df["t"] < df["t"].max())
    asset_growth = df["acct_010"] / g["acct_010"].shift(1) - 1
    contiguous = g["t"].shift(1) == df["t"] - 1
    exits_by_state_t = (
        df.loc[df["exited_next_q"]].assign(t=lambda x: x["t"] + 1).groupby(["state", "t"]).size()
    )
    has_exit = pd.MultiIndex.from_frame(df[["state", "t"]]).isin(exits_by_state_t.index)
    m = contiguous & (asset_growth > 0.25) & has_exit & (df["acct_010"] > 5e7)
    flags["f_suspected_merger"] = log(m, "assets +25% QoQ in a quarter when a same-state CU exited", "acct_010")

    # 3. Robust z-score of quarter-over-quarter change in each delinquency rate,
    #    computed cross-sectionally within each quarter (median / MAD).
    for p in ["total", "card", "auto", "re"]:
        col = f"dq_rate_{p}"
        chg = df[col] - g[col].shift(1)
        size_ok = df[f"bal_{p}"] > 1e7  # ignore tiny portfolios where one loan swings the rate
        med = chg.groupby(df["t"]).transform("median")
        mad = (chg - med).abs().groupby(df["t"]).transform("median")
        # floor MAD at 25bp so quarters where most CUs show ~0 change do not explode z
        z = 0.6745 * (chg - med) / mad.clip(lower=0.0025)
        m = size_ok & contiguous & (z.abs() > 6)
        if p == "re":  # the change into 2022Q1 crosses the definition break, not a data error
            m = m & (df["t"] != RE_BREAK_T)
        flags[f"f_outlier_dq_{p}"] = log(m, "robust z of QoQ change in 60+ day delinquency rate > 6 (MAD floored at 25bp)", col)

    flags = flags.fillna(False).astype(bool)
    df = pd.concat([df, flags], axis=1)
    df["any_flag"] = flags.any(axis=1)
    log_df = pd.concat(logs, ignore_index=True)
    return df, log_df


def add_macro(df: pd.DataFrame) -> pd.DataFrame:
    fred = pd.read_csv(RAW / "fred_quarterly.csv").rename(columns={"quarter": "period"})
    return df.merge(fred, on="period", how="left")


def main():
    df = load_raw()
    df = splice_real_estate(df)
    df = decumulate(df)
    df = build_measures(df)
    df, anomalies = flag_anomalies(df)
    df = add_macro(df)
    df.to_parquet(OUT / "panel.parquet", index=False)
    anomalies.to_csv(OUT / "anomaly_log.csv", index=False)

    summary = anomalies.groupby("rule").agg(records=("cu_number", "size"), credit_unions=("cu_number", "nunique"))
    print(f"panel rows {len(df):,}  credit unions {df.cu_number.nunique():,}  quarters {df.period.nunique()}")
    print(f"rows with any flag {df.any_flag.sum():,} ({df.any_flag.mean():.2%})")
    print(summary.to_string())


if __name__ == "__main__":
    main()
