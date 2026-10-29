"""Figures for the README (written to figures/)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"total": "#2B5C8A", "card": "#C8553D", "auto": "#5B8C5A", "re": "#8A6FB0"}


def main():
    ind = pd.read_csv(ROOT / "powerbi" / "industry_quarterly.csv")
    x = range(len(ind))
    ticks = [i for i, p in enumerate(ind.period) if p.endswith("Q1")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for p in ["total", "card", "auto", "re"]:
        axes[0].plot(x, ind[f"dq_rate_{p}"] * 100, label=p, color=COLORS[p])
        axes[1].plot(x, ind[f"nco_rate_{p}"] * 100, label=p, color=COLORS[p])
    axes[0].set_title("60+ day delinquency rate, all credit unions (%)")
    axes[1].set_title("Annualized net charge-off rate (%)")
    brk = list(ind.period).index("2022Q1")
    for ax in axes:
        ax.axvline(brk, color="grey", lw=0.8, ls=":")
        ax.text(brk, ax.get_ylim()[1] * 0.98, " 2022 form redesign\n (affects re)", fontsize=7, color="grey", va="top")
        ax.set_xticks(ticks, [ind.period[i][:4] for i in ticks], rotation=0)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "industry_credit_trends.png", dpi=150)

    bt = pd.read_csv(ROOT / "outputs" / "origination_backtest.csv")
    s = pd.read_csv(ROOT / "outputs" / "origination_industry_series.csv")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(s.period, s.granted / 1e9, color="#333333", label="actual")
    for m, c in [("sarimax_macro", "#2B5C8A"), ("seasonal_naive", "#C8553D")]:
        b = bt[bt.model == m]
        ax.plot(b.period, b.forecast / 1e9, "o--", ms=3, color=c, label=f"{m} (1-quarter-ahead)")
    ax.axvline("2024Q3", color="grey", lw=0.8, ls=":")
    ax.text("2024Q3", ax.get_ylim()[1] * 0.97, " test window", va="top", fontsize=8, color="grey")
    ax.set_title("Quarterly loans granted, all federally insured credit unions ($B)")
    ax.set_xticks([p for p in s.period if p.endswith("Q1")], [p[:4] for p in s.period if p.endswith("Q1")])
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "origination_backtest.png", dpi=150)
    print("figures written")


if __name__ == "__main__":
    main()
