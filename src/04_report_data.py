"""CRISP-DM phase 6 (Deployment): reporting outputs.

1. reports/portfolio_risk_report.rdl
   Paginated report for Microsoft Report Builder (SSRS). Data is embedded in
   the report as XML (XML data source, blank connection string), so it opens
   and runs without a database connection. Peer group: the 25 largest credit
   unions by total assets at 2026Q2. Shows 60+ day delinquency and annualized
   net charge-off rates for the last 8 quarters plus the 2026Q3 forecast, with
   a Portfolio parameter (Total loans / Credit card / Auto / Real estate).

2. powerbi/*.csv
   Flat tables for a Power BI dashboard: industry quarterly trends, large
   credit union panel, forecasts, backtest scores, anomaly log summary.
"""
import uuid
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REP = ROOT / "reports"
PBI = ROOT / "powerbi"
REP.mkdir(exist_ok=True)
PBI.mkdir(exist_ok=True)
PORT_LABEL = {"total": "Total loans", "card": "Credit card", "auto": "Auto", "re": "Real estate"}


def title_case(name):
    return " ".join(w.capitalize() if len(w) > 3 else w for w in str(name).replace("'", "").split())


def build_report_rows(df, fc):
    last_t = df["t"].max()
    peers = df[df["t"] == last_t].nlargest(25, "acct_010")[["cu_number", "cu_name", "state"]].reset_index(drop=True)
    peers["rank"] = peers.index + 1
    hist = df[(df["t"] > last_t - 8) & df["cu_number"].isin(peers["cu_number"])]
    rows = []
    for port in PORT_LABEL:
        for _, r in hist.iterrows():
            rows.append({"cu_number": r.cu_number, "Period": r.period, "Portfolio": PORT_LABEL[port],
                         "Kind": "Actual", "Balance": r[f"bal_{port}"],
                         "DQ": r[f"dq_rate_{port}"], "NCO": r[f"nco_rate_{port}"]})
        f = fc[fc["portfolio"] == port].pivot_table(index="cu_number", columns="target", values="forecast_2026Q3")
        for cu in peers["cu_number"]:
            rows.append({"cu_number": cu, "Period": "2026Q3 fcst", "Portfolio": PORT_LABEL[port], "Kind": "Forecast",
                         "Balance": np.nan,
                         "DQ": f.loc[cu, "dq"] if cu in f.index else np.nan,
                         "NCO": f.loc[cu, "nco"] if cu in f.index else np.nan})
    out = pd.DataFrame(rows).merge(peers, on="cu_number")
    out["CU"] = out.apply(lambda r: f"{r['rank']:02d}. {title_case(r.cu_name)} ({r.state})", axis=1)
    out["Rank"] = out["rank"].map(lambda x: f"{x:02d}")
    return out, peers


def xml_rows(rows):
    parts = []
    for _, r in rows.iterrows():
        attrs = {"CU": r.CU, "Rank": r.Rank, "Period": r.Period, "Portfolio": r.Portfolio, "Kind": r.Kind}
        for k in ["Balance", "DQ", "NCO"]:
            if pd.notna(r[k]):
                attrs[k] = f"{r[k]:.6f}"
        parts.append("<R " + " ".join(f"{k}={quoteattr(str(v))}" for k, v in attrs.items()) + " />")
    return "<Rows>" + "".join(parts) + "</Rows>"


def textbox(name, value, fmt=None, bold=False, size="8pt", bg=None, align=None, width_hint=None, color=None):
    style = f"<FontSize>{size}</FontSize>"
    if bold:
        style += "<FontWeight>Bold</FontWeight>"
    if fmt:
        style += f"<Format>{fmt}</Format>"
    if color:
        style += f"<Color>{color}</Color>"
    para_style = f"<TextAlign>{align}</TextAlign>" if align else ""
    box_style = ("<Border><Color>LightGrey</Color><Style>Solid</Style></Border>"
                 "<PaddingLeft>2pt</PaddingLeft><PaddingRight>2pt</PaddingRight>"
                 "<PaddingTop>1pt</PaddingTop><PaddingBottom>1pt</PaddingBottom>")
    if bg:
        box_style += f"<BackgroundColor>{bg}</BackgroundColor>"
    return (f'<Textbox Name="{name}"><CanGrow>true</CanGrow><KeepTogether>true</KeepTogether>'
            f"<Paragraphs><Paragraph><TextRuns><TextRun><Value>{escape(value)}</Value>"
            f"<Style>{style}</Style></TextRun></TextRuns><Style>{para_style}</Style></Paragraph></Paragraphs>"
            f"<Style>{box_style}</Style></Textbox>")


def matrix(name, measure, top, title):
    fc_bg = '=IIf(First(Fields!Kind.Value)="Forecast","#FFF4CE","White")'
    value = (f"=IIf(Count(Fields!{measure}.Value)=0, Nothing, Sum(CDbl(Fields!{measure}.Value)))")
    return f"""
<Textbox Name="{name}Title"><CanGrow>true</CanGrow><KeepTogether>true</KeepTogether>
<Paragraphs><Paragraph><TextRuns><TextRun><Value>{escape(title)}</Value><Style><FontSize>10pt</FontSize><FontWeight>Bold</FontWeight></Style></TextRun></TextRuns><Style /></Paragraph></Paragraphs>
<Top>{top}in</Top><Left>0in</Left><Height>0.25in</Height><Width>10in</Width><Style /></Textbox>
<Tablix Name="{name}">
<TablixCorner><TablixCornerRows><TablixCornerRow><TablixCornerCell><CellContents>
{textbox(name + "Corner", "Credit union (rank by assets)", bold=True, bg="#E7EEF7")}
</CellContents></TablixCornerCell></TablixCornerRow></TablixCornerRows></TablixCorner>
<TablixBody>
<TablixColumns><TablixColumn><Width>0.78in</Width></TablixColumn></TablixColumns>
<TablixRows><TablixRow><Height>0.2in</Height><TablixCells><TablixCell><CellContents>
{textbox(name + "Value", value, fmt="0.00%", align="Right", bg=fc_bg)}
</CellContents></TablixCell></TablixCells></TablixRow></TablixRows>
</TablixBody>
<TablixColumnHierarchy><TablixMembers><TablixMember>
<Group Name="{name}Period"><GroupExpressions><GroupExpression>=Fields!Period.Value</GroupExpression></GroupExpressions></Group>
<SortExpressions><SortExpression><Value>=Fields!Period.Value</Value></SortExpression></SortExpressions>
<TablixHeader><Size>0.25in</Size><CellContents>
{textbox(name + "PeriodHdr", "=Fields!Period.Value", bold=True, align="Right", bg="#E7EEF7")}
</CellContents></TablixHeader>
</TablixMember></TablixMembers></TablixColumnHierarchy>
<TablixRowHierarchy><TablixMembers><TablixMember>
<Group Name="{name}CU"><GroupExpressions><GroupExpression>=Fields!CU.Value</GroupExpression></GroupExpressions></Group>
<SortExpressions><SortExpression><Value>=Fields!Rank.Value</Value></SortExpression></SortExpressions>
<TablixHeader><Size>2.9in</Size><CellContents>
{textbox(name + "CUHdr", "=Fields!CU.Value")}
</CellContents></TablixHeader>
</TablixMember></TablixMembers></TablixRowHierarchy>
<DataSetName>RiskData</DataSetName>
<Top>{top + 0.3}in</Top><Left>0in</Left><Height>0.45in</Height><Width>3.68in</Width>
<Style><Border><Style>None</Style></Border></Style>
</Tablix>"""


def build_rdl(rows, origination, scores):
    query = "<Query><XmlData>" + xml_rows(rows) + "</XmlData><ElementPath>Rows {}/R</ElementPath></Query>"
    fields = "".join(f'<Field Name="{f}"><DataField>{f}</DataField><rd:TypeName>System.String</rd:TypeName></Field>'
                     for f in ["CU", "Rank", "Period", "Portfolio", "Kind", "Balance", "DQ", "NCO"])
    params = "".join(f"<ParameterValue><Value>{v}</Value><Label>{v}</Label></ParameterValue>" for v in PORT_LABEL.values())

    o = origination.iloc[0]
    sc = scores[(scores.window == "test")].set_index(["target", "portfolio", "model"])["wMAE_bp"]
    note = (
        "Source: NCUA 5300 Call Report quarterly data, 2015Q2-2026Q2. Delinquency = 60+ days past due balance / portfolio "
        "balance. Net charge-off rate = quarterly charge-offs less recoveries, annualized, over quarter-end balance. "
        "Shaded column = one-quarter-ahead forecast from a balance-weighted LightGBM model chosen on a 2022Q3-2024Q2 "
        "validation window. Test window 2024Q3-2026Q2, balance-weighted MAE (model vs. last-quarter persistence): "
        f"total-loan delinquency {sc[('dq','total','lgbm')]:.1f} vs {sc[('dq','total','persistence')]:.1f} bp; "
        f"total net charge-offs {sc[('nco','total','lgbm')]:.1f} vs {sc[('nco','total','persistence')]:.1f} bp. "
        f"Industry loans granted, 2026Q3 forecast (SARIMAX with macro drivers): ${o.forecast_usd/1e9:.1f}B "
        f"(80% interval ${o.pi80_low_usd/1e9:.1f}B-${o.pi80_high_usd/1e9:.1f}B). Blank cells: portfolio under $10M "
        "or outside model scope."
    )
    m1 = matrix("DQMatrix", "DQ", 0.75, "60+ day delinquency rate")
    m2 = matrix("NCOMatrix", "NCO", 1.75, "Annualized net charge-off rate")
    ds_id, rep_id = uuid.uuid4(), uuid.uuid4()
    return f"""<?xml version="1.0" encoding="utf-8"?>
<Report xmlns="http://schemas.microsoft.com/sqlserver/reporting/2016/01/reportdefinition" xmlns:rd="http://schemas.microsoft.com/SQLServer/reporting/reportdesigner">
<AutoRefresh>0</AutoRefresh>
<DataSources><DataSource Name="EmbeddedXml"><ConnectionProperties><DataProvider>XML</DataProvider><ConnectString /></ConnectionProperties><rd:SecurityType>None</rd:SecurityType><rd:DataSourceID>{ds_id}</rd:DataSourceID></DataSource></DataSources>
<DataSets><DataSet Name="RiskData"><Query><DataSourceName>EmbeddedXml</DataSourceName><CommandText>{escape(query)}</CommandText></Query>
<Fields>{fields}</Fields>
<Filters><Filter><FilterExpression>=Fields!Portfolio.Value</FilterExpression><Operator>Equal</Operator><FilterValues><FilterValue>=Parameters!Portfolio.Value</FilterValue></FilterValues></Filter></Filters>
</DataSet></DataSets>
<ReportSections><ReportSection><Body><ReportItems>
<Textbox Name="ReportTitle"><CanGrow>true</CanGrow><KeepTogether>true</KeepTogether>
<Paragraphs><Paragraph><TextRuns><TextRun><Value>="Credit Union Loan Portfolio Risk Report: " &amp; Parameters!Portfolio.Value</Value><Style><FontSize>14pt</FontSize><FontWeight>Bold</FontWeight></Style></TextRun></TextRuns><Style /></Paragraph></Paragraphs>
<Top>0in</Top><Left>0in</Left><Height>0.35in</Height><Width>10in</Width><Style /></Textbox>
<Textbox Name="Subtitle"><CanGrow>true</CanGrow><KeepTogether>true</KeepTogether>
<Paragraphs><Paragraph><TextRuns><TextRun><Value>25 largest federally insured credit unions by total assets (2026Q2). Last 8 quarters actual, plus 2026Q3 forecast (shaded).</Value><Style><FontSize>8pt</FontSize><Color>DimGray</Color></Style></TextRun></TextRuns><Style /></Paragraph></Paragraphs>
<Top>0.38in</Top><Left>0in</Left><Height>0.25in</Height><Width>10in</Width><Style /></Textbox>
{m1}
{m2}
<Textbox Name="Notes"><CanGrow>true</CanGrow><KeepTogether>true</KeepTogether>
<Paragraphs><Paragraph><TextRuns><TextRun><Value>{escape(note)}</Value><Style><FontSize>7pt</FontSize><Color>DimGray</Color></Style></TextRun></TextRuns><Style /></Paragraph></Paragraphs>
<Top>2.75in</Top><Left>0in</Left><Height>0.6in</Height><Width>10in</Width><Style /></Textbox>
</ReportItems><Height>3.4in</Height><Style /></Body>
<Width>10in</Width>
<Page><PageHeight>8.5in</PageHeight><PageWidth>11in</PageWidth><LeftMargin>0.5in</LeftMargin><RightMargin>0.5in</RightMargin><TopMargin>0.5in</TopMargin><BottomMargin>0.5in</BottomMargin><Style /></Page>
</ReportSection></ReportSections>
<ReportParameters><ReportParameter Name="Portfolio"><DataType>String</DataType>
<DefaultValue><Values><Value>Total loans</Value></Values></DefaultValue><Prompt>Portfolio</Prompt>
<ValidValues><ParameterValues>{params}</ParameterValues></ValidValues></ReportParameter></ReportParameters>
<ReportParametersLayout><GridLayoutDefinition><NumberOfColumns>4</NumberOfColumns><NumberOfRows>1</NumberOfRows>
<CellDefinitions><CellDefinition><ColumnIndex>0</ColumnIndex><RowIndex>0</RowIndex><ParameterName>Portfolio</ParameterName></CellDefinition></CellDefinitions>
</GridLayoutDefinition></ReportParametersLayout>
<rd:ReportUnitType>Inch</rd:ReportUnitType>
<rd:ReportID>{rep_id}</rd:ReportID>
</Report>
"""


def power_bi_tables(df, fc, peers):
    ports = list(PORT_LABEL)
    agg = {"cu_count": ("cu_number", "nunique"), "assets": ("acct_010", "sum"), "members": ("acct_083", "sum"),
           "loans_granted": ("q_granted_amt", "sum")}
    for p in ports:
        agg[f"bal_{p}"] = (f"bal_{p}", "sum")
        agg[f"dq_{p}"] = (f"dq_{p}", "sum")
        agg[f"nco_{p}"] = (f"q_nco_{p}", "sum")
    ind = df.groupby(["t", "period"], as_index=False).agg(**agg)
    for p in ports:
        ind[f"dq_rate_{p}"] = ind[f"dq_{p}"] / ind[f"bal_{p}"]
        ind[f"nco_rate_{p}"] = 4 * ind[f"nco_{p}"] / ind[f"bal_{p}"]
    ind.loc[ind["t"] < 2, ["loans_granted"] + [f"nco_{p}" for p in ports] + [f"nco_rate_{p}" for p in ports]] = np.nan
    ind.drop(columns="t").to_csv(PBI / "industry_quarterly.csv", index=False)

    big = df[df.groupby("cu_number")["acct_010"].transform("max") >= 1e9]
    keep = ["cu_number", "cu_name", "state", "period", "acct_010", "q_granted_amt"] + \
           [f"bal_{p}" for p in ports] + [f"dq_rate_{p}" for p in ports] + [f"nco_rate_{p}" for p in ports] + ["any_flag"]
    big[keep].rename(columns={"acct_010": "total_assets", "q_granted_amt": "loans_granted_q"}).to_csv(
        PBI / "cu_quarterly_over_1b.csv", index=False)
    fc.to_csv(PBI / "cu_forecast_2026Q3.csv", index=False)
    pd.read_csv(OUT / "credit_risk_backtest_scores.csv").to_csv(PBI / "credit_risk_backtest_scores.csv", index=False)
    pd.read_csv(OUT / "origination_backtest.csv").to_csv(PBI / "origination_backtest.csv", index=False)
    an = pd.read_csv(ROOT / "data" / "processed" / "anomaly_log.csv")
    an.groupby(["period", "rule"]).size().rename("records").reset_index().to_csv(PBI / "anomaly_summary.csv", index=False)


def main():
    df = pd.read_parquet(ROOT / "data" / "processed" / "panel.parquet")
    fc = pd.read_csv(OUT / "credit_risk_forecast_2026Q3_by_cu.csv")
    origination = pd.read_csv(OUT / "origination_forecast_2026Q3.csv")
    scores = pd.read_csv(OUT / "credit_risk_backtest_scores.csv")
    rows, peers = build_report_rows(df, fc)
    rows.to_csv(REP / "portfolio_risk_report_data.csv", index=False)
    (REP / "portfolio_risk_report.rdl").write_text(build_rdl(rows, origination, scores), encoding="utf-8")
    power_bi_tables(df, fc, peers)
    print(f"report rows {len(rows)}; peers {len(peers)}; rdl {(REP / 'portfolio_risk_report.rdl').stat().st_size/1e3:.0f} KB")


if __name__ == "__main__":
    main()
