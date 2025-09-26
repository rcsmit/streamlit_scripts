import pandas as pd
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import r2_score
from typing import List, Tuple
import random

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_scatterplot(
    df_temp: pd.DataFrame,
    what_to_show_l,
    what_to_show_r,
    aggregate_duplicates: bool = True,
    trend_on_aggregated: bool = True,
) -> None:
    """Draw a scatter with optional point aggregation by (x, y) frequency.

    Parameters
    ----------
    df_temp : DataFrame
        Source data
    what_to_show_l : str | List[str]
        Column name for x or list of one
    what_to_show_r : str | List[str]
        Column name for y or list of one
    aggregate_duplicates : bool
        If True, aggregate identical (x, y) pairs and size points by frequency
    trend_on_aggregated : bool
        If True, fit OLS trend on aggregated data; else on raw data
    """

    # Keep your original conversions and polyfit lines, but make them robust.
    what_to_show_l = (
        what_to_show_l if isinstance(what_to_show_l, list) else [what_to_show_l]
    )
    what_to_show_r = (
        what_to_show_r if isinstance(what_to_show_r, list) else [what_to_show_r]
    )

    x_series = pd.to_numeric(df_temp[what_to_show_l[0]], errors="coerce")
    y_series = pd.to_numeric(df_temp[what_to_show_r[0]], errors="coerce")

    # Align and drop NaNs for regression
    valid = x_series.notna() & y_series.notna()
    x_vals = x_series[valid].to_numpy()
    y_vals = y_series[valid].to_numpy()

    if len(x_vals) >= 2:
        m, b = np.polyfit(x_vals, y_vals, 1)
        predict = np.poly1d([m, b])
        r2 = r2_score(y_vals, predict(x_vals))
    else:
        m, b, r2 = np.nan, np.nan, np.nan

    # De kolom 'R square' is een zogenaamde goodness-of-fit maat.
    # Deze maat geeft uitdrukking aan hoe goed de geobserveerde data clusteren rond de geschatte regressielijn.
    # In een enkelvoudige lineaire regressie is dat het kwadraat van de correlatie.
    # De proportie wordt meestal in een percentage ‘verklaarde variantie’ uitgedrukt.
    #  In dit voorbeeld betekent R square dus dat de totale variatie in vetpercentages voor 66% verklaard
    #    kan worden door de lineaire regressie c.q. de verschillen in leeftijd.
    # https://wikistatistiek.amc.nl/index.php/Lineaire_regressie

    # print (r2)
    # m, b = np.polyfit(x_, y_, 1)
    # print (m,b)

    # -----------------------------
    # Plot (keeping your commented original block)
    # -----------------------------

    # fig1xyz = px.scatter(df_temp, x=what_to_show_l[0], y=what_to_show_r[0],
    #                     trendline="ols", trendline_scope = 'overall',trendline_color_override = 'black'
    #         )

    if aggregate_duplicates:
        # Count duplicates per (x,y)
        df_temp2 = (
            df_temp.groupby([what_to_show_l[0], what_to_show_r[0]])
            .size()
            .reset_index(name="count")
        )
        plot_df = df_temp2
        size_kw = {"size": "count"}
    else:
        plot_df = df_temp.copy()
        size_kw = {}

    trend_scope = "overall"
    if not trend_on_aggregated:
        # compute trend on raw points, but still plot aggregated sizes
        trend_data = df_temp[[what_to_show_l[0], what_to_show_r[0]]].dropna()
        fig_raw = px.scatter(
            trend_data,
            x=what_to_show_l[0],
            y=what_to_show_r[0],
            trendline="ols",
            trendline_scope=trend_scope,
            trendline_color_override="black",
        )
        # Extract trendline results to re-use the fitted line in fig; simplest is to just let PX compute again on plot_df
        # For simplicity and stability we keep one call with trendline based on plot_df below.

    fig1xyz = px.scatter(
        plot_df,
        x=what_to_show_l[0],
        y=what_to_show_r[0],
        trendline="ols",
        trendline_scope=trend_scope,
        trendline_color_override="black",
        **size_kw,
    )

    correlation_sp = round(x_series.corr(y_series, method="spearman"), 3)
    correlation_p = round(x_series.corr(y_series, method="pearson"), 3)

    title_scatter_plotly = (
        f"{what_to_show_l[0]} -  {what_to_show_r[0]}<br>"
        f"Correlation spearman = {correlation_sp} - Correlation pearson = {correlation_p}"
        f"<br>y = {np.round(m,2)}*x + {np.round(b,2)} | r2 = {np.round(r2,4)}"
    )

    fig1xyz.update_layout(
        title=dict(
            text=title_scatter_plotly,
            # x=0.5,
            # y=0.95,
            font=dict(
                family="Arial",
                size=12,
                color="#000000",
            ),
        ),
        xaxis_title=what_to_show_l[0],
        yaxis_title=what_to_show_r[0],
    )

    st.plotly_chart(fig1xyz, use_container_width=True)


def clean_staffmembernr_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading 'EC0' from staffmembernr and convert to nullable Int64."""
    df = df.copy()
    df["staffmembernr"] = (
        df["staffmembernr"].astype(str).str.replace(r"^EC0", "", regex=True)
    )
    df["staffmembernr"] = pd.to_numeric(df["staffmembernr"], errors="coerce").astype(
        "Int64"
    )
    return df


def summarize_ratings(df: pd.DataFrame, grouper: str, what: str) -> pd.DataFrame:
    """Return mean, review_count, and percentage distribution for a rating column.

    If `what == 'nps'`, percentages are for 1..10, else for 1..5.
    """
    mean_col = df.groupby(grouper, dropna=False)[what].mean().rename(f"mean_{what}")
    review_count = (
        df.groupby(grouper, dropna=False)[what].count().rename("review_count")
    )

    if what == "nps":
        review_range = range(0, 11)
    else:
        review_range = range(1, 6)

    counts = (
        df.pivot_table(index=grouper, columns=what, values="datum", aggfunc="size")
        .reindex(columns=review_range, fill_value=0)
        .fillna(0)
    )
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    percentages = percentages.rename(
        columns={i: f"percentage_{i}" for i in review_range}
    )

    result = pd.concat([mean_col, review_count, percentages], axis=1).reset_index()
    cols_to_round = [f"mean_{what}"] + [f"percentage_{i}" for i in review_range]
    result[cols_to_round] = result[cols_to_round].round(2)
    return result


def summarize_ratings_all_columns(df: pd.DataFrame, grouper: str) -> pd.DataFrame:
    """Return mean, review_count, and percentage distribution for a rating column.

    If `what == 'nps'`, percentages are for 1..10, else for 1..5.
    """
    mean_col = (
        df.groupby(grouper, dropna=False)[
            [
                "nps",
                "Instructions",
                "Connection",
                "Adjustments",
                "Tangibles",
                "Knowledge",
            ]
        ]
        .mean(numeric_only=True)
        .round(1)
        .add_prefix("mean_")
    )
    review_count = (
        df.groupby(grouper, dropna=False)[["Connection"]]
        .count()
        .rename(columns={"Connection": "review_count"})
    )

    result = pd.concat([mean_col, review_count], axis=1).reset_index()
    # cols_to_round = [f"mean_{what}"] + [f"percentage_{i}" for i in review_range]
    # result[cols_to_round] = result[cols_to_round].round(2)
    return result


def compute_nps(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Compute NPS per group.
    NPS = %Promoters (9–10) - %Detractors (0–6)
    Returns a DataFrame with columns group_cols + ['NPS','pct_promoters','pct_detractors','responses']
    """
    tmp = df.copy()
    tmp["nps"] = pd.to_numeric(tmp["nps"], errors="coerce")
    tmp = tmp.dropna(subset=["nps"])  # only valid NPS rows

    # categorize
    tmp["nps_cat"] = pd.cut(
        tmp["nps"], bins=[-0.5, 6, 8, 10], labels=["detractor", "passive", "promoter"]
    )

    # counts per group and cat
    counts = (
        tmp.groupby(group_cols + ["nps_cat"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    totals = counts.sum(axis=1)
    pct_prom = counts.get("promoter", 0).div(totals).mul(100)
    pct_det = counts.get("detractor", 0).div(totals).mul(100)

    out = pd.DataFrame(
        {
            "NPS": (pct_prom - pct_det).round(2),
            "pct_promoters": pct_prom.round(2),
            "pct_detractors": pct_det.round(2),
            "responses": totals.astype(int),
        }
    ).reset_index()
    return out


def plot_nps_stack(df: pd.DataFrame, x_col: str):
    """Plot stacked bar chart of NPS distribution with responses on secondary axis.
    Args:
        df (pd.DataFrame): DataFrame with columns: x_col, pct_promoters, pct_detractors, responses
        x_col (str): column name for x-axis
    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure
    """
    # Expect columns: x_col, pct_promoters, pct_detractors, responses
    d = df.copy()
    d["pct_passives"] = 100 - d["pct_promoters"] - d["pct_detractors"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Stacked bars
    fig.add_bar(
        x=d[x_col],
        y=d["pct_detractors"],
        name="Detractors (1-6)",
        marker_color="#c0392b",
        hovertemplate="%{y:.1f}% detractors<extra></extra>",
    )
    fig.add_bar(
        x=d[x_col],
        y=d["pct_passives"],
        name="Passives (7-8)",
        marker_color="#f1c40f",
        hovertemplate="%{y:.1f}% passives<extra></extra>",
    )
    fig.add_bar(
        x=d[x_col],
        y=d["pct_promoters"],
        name="Promoters (9-10)",
        marker_color="#27ae60",
        hovertemplate="%{y:.1f}% promoters<extra></extra>",
    )

    # Responses on secondary axis
    fig.add_trace(
        go.Scatter(
            x=d[x_col],
            y=d["responses"],
            mode="lines+markers+text",
            text=[str(v) for v in d["responses"]],
            textposition="top center",
            line=dict(color="black", width=2),
            marker=dict(size=6),
            name="Responses",
            hovertemplate="%{y} responses<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        barmode="stack",
        title="NPS distribution and responses",
        hovermode="x unified",
        legend_title_text="Category",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_yaxes(title_text="Percent", range=[0, 100], secondary_y=False)
    fig.update_yaxes(title_text="Responses", secondary_y=True)
    fig.update_xaxes(title_text=x_col)

    return fig


def plot_nps_per_branche(result: pd.DataFrame):
    """Plot NPS per branche
    Args:
        result (pd.DataFrame): DataFrame with columns: branche, mean_nps, review_count, percentage_1..percentage_10
    """
    # 1) Groepen uit de percentage-kolommen
    result["pct_detractors"] = result[[f"percentage_{i}" for i in range(0, 7)]].sum(
        axis=1
    )
    result["pct_passives"] = result[["percentage_7", "percentage_8"]].sum(axis=1)
    result["pct_promoters"] = result[["percentage_9", "percentage_10"]].sum(axis=1)
    result["NPS_calc"] = result["pct_promoters"] - result["pct_detractors"]
    # 2) Sorteervolgorde (optioneel: op mean_nps)
    order = result.sort_values("mean_nps")["branche"].astype(str).tolist()
    dfp = result.set_index("branche").loc[order].reset_index()

    # 3) Gestapelde balken
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Detractors (1–6)",
            x=dfp["branche"],
            y=dfp["pct_detractors"],
            marker_color="red",
            text=dfp["pct_detractors"].round(0).astype(int),
            texttemplate="%{text}%",
            textposition="inside",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Passives (7–8)",
            x=dfp["branche"],
            y=dfp["pct_passives"],
            marker_color="orange",
            text=dfp["pct_passives"].round(0).astype(int),
            texttemplate="%{text}%",
            textposition="inside",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Promoters (9–10)",
            x=dfp["branche"],
            y=dfp["pct_promoters"],
            marker_color="green",
            text=dfp["pct_promoters"].round(0).astype(int),
            texttemplate="%{text}%",
            textposition="inside",
        )
    )

    # 4) Layout en annotaties (n en mean)
    fig.update_layout(
        barmode="stack",
        xaxis=dict(title=""),
        yaxis=dict(range=[0, 110], title="Percent", ticksuffix="%"),
        legend_title_text="Categorie",
        title="NPS distribution per branche",
        bargap=0.15,
    )

    for _, r in dfp.iterrows():
        fig.add_annotation(
            x=r["branche"],
            y=107,
            text=f"n={int(r['review_count'])}<br>mean={r['mean_nps']:.1f}<br>NPS={r['NPS_calc']:.1f}",
            showarrow=False,
            font=dict(size=10),
        )

    # Zorg dat de x-orde exact zo blijft
    fig.update_xaxes(categoryorder="array", categoryarray=order)

    st.plotly_chart(fig, use_container_width=True)


def nps_per_period(df: pd.DataFrame):
    """Make lineplots of NPS per period (overall and by branche if available).
    Args:
        df (pd.DataFrame): df with the survey results
    """
    nps_period = compute_nps(df, ["year_period"]).sort_values("year_period")
    st.subheader("NPS per period (overall)")
    # st.write(nps_period)
    fig_nps_overall = px.line(
        nps_period,
        x="year_period",
        y="NPS",
        markers=True,
        title="NPS per period (overall)",
    )
    fig_nps_overall.update_layout(xaxis_title="period", yaxis_title="NPS")
    st.plotly_chart(fig_nps_overall, use_container_width=True)

    fig = plot_nps_stack(nps_period, x_col="year_period")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # NPS per period by branche
    # -----------------------------
    if "branche" in df.columns:
        nps_period_acc = compute_nps(df, ["year_period", "branche"]).sort_values(
            ["year_period", "branche"]
        )
        st.subheader("NPS per period by branche")
        # st.write(nps_period_acc)
        fig_nps_acc = px.line(
            nps_period_acc,
            x="year_period",
            y="NPS",
            color="branche",
            markers=True,
            title="NPS per period by branche",
        )
        fig_nps_acc.update_layout(
            xaxis_title="period", yaxis_title="NPS", legend_title="branche"
        )
        st.plotly_chart(fig_nps_acc, use_container_width=True)
        st.write("Double click on an branche in the legenda to see a correct line")


def Aftersales_score_per_period(df: pd.DataFrame):
    """Make a bar plot of the mean Aftersales score per period.
    Args:
        df (pd.DataFrame): df with the survey results
    """
    st.write(df)
    # Mean Aftersales per period (kept)
    period_avg = df.groupby(["year_period"])["Aftersales"].mean().reset_index()

    # st.write(period_avg)
    # Plot with Plotly
    fig = px.bar(
        period_avg,
        x="year_period",
        y="Aftersales",
        title=f"Average Aftersales Score per period",
    )

    fig.update_layout(
        xaxis_title="period",
        yaxis_title="Average Aftersales Score",
    )
    st.plotly_chart(fig)


def ols_corr(df):
    """Make an multiple lineair regression analyses (OLS)
    Args:
        df (dataframe): dataframe with the survey results
    """
    # Select predictors and target
    X = df[["Connection", "Adjustments", "Tangibles", "Knowledge"]]
    y = df["Instructions"]
    # y = df["nps"]

    # Drop rows with missing values
    data = pd.concat([X, y], axis=1).dropna()

    # Your original switch to predict Instructions from the others kept as-is
    X = data[["Connection", "Adjustments", "Tangibles", "Knowledge"]]
    y = data["Instructions"]
    # y=data["nps"]

    # Add constant for regression
    X = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y, X).fit()
    st.write(model.summary())
    st.write(
        df[
            [
                "nps",
                "Instructions",
                "Connection",
                "Adjustments",
                "Tangibles",
                "Knowledge",
            ]
        ].corr()
    )


def reviews_per_period(df: pd.DataFrame):
    """Make a lineplot of the number of reviews per period (total and per branche)

    Args:
        df (pd.DataFrame): df with the survey results
    """
    # Reviews per period (count NPS responses)
    reviews_per_period = (
        df[df["nps"].notna()].groupby("year_period").size().reset_index(name="reviews")
    )
    fig = px.line(
        reviews_per_period, x="year_period", y="reviews", title="Reviews per period"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optional: reviews per period by branche
    reviews_per_period_type = (
        df[df["nps"].notna()]
        .groupby(["year_period", "branche"])
        .size()
        .reset_index(name="reviews")
        .sort_values(["branche", "year_period"])
    )
    fig2 = px.line(
        reviews_per_period_type,
        x="year_period",
        y="reviews",
        color="branche",
        markers=True,
        title="Reviews per period by branche",
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.write("Double click on an branche in the legenda to see a correct line")


def plot_indicator_evolution(
    df: pd.DataFrame, period_col: str = "year_period", score_col: str = "Aftersales"
):
    """Plot the evolution of a score column over time (period_col).
    Shows stacked bar chart of % distribution per score (1–5 or 0–10) per period,
    with average line on secondary axis.
    Args:
        df (pd.DataFrame): DataFrame with the survey results
        period_col (str, optional): Column with the period (e.g. 'year_period'). Defaults to "year_period".
        score_col (str, optional): Column with the score (e.g. 'nps' or 'Instructions'). Defaults to "Aftersales".
    """

    # 1) Prepare % distribution 1–5 per period
    show_counts = st.checkbox(
        "Show counts instead of %", value=False, key=f"counts-{score_col}"
    )

    if score_col == "nps":
        rev_max = 10
        rev_min = 0
    else:
        rev_min = 1
        rev_max = 5
    review_range = range(rev_min, rev_max)
    temp = df[[period_col, score_col]].dropna().copy()
    temp[score_col] = (
        pd.to_numeric(temp[score_col], errors="coerce")
        .round()
        .clip(rev_min, rev_max)
        .astype(int)
    )

    counts = (
        temp.groupby([period_col, score_col])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=review_range, fill_value=0)
    )

    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    pct = pct.reset_index().sort_values(period_col)

    # 2) Mean per period
    means = (
        temp.groupby(period_col)[score_col].mean().reset_index().sort_values(period_col)
    )

    # 3) Build figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x_vals = pct[period_col].tolist()

    # Colors close to the screenshot
    color_map = {
        0: "#903900",  # Very unsatisfied
        1: "#c0392b",  # Very unsatisfied
        2: "#e67e22",  # Unsatisfied
        3: "#f1c40f",  # Neutral-ish
        4: "#2ecc71",  # Satisfied
        5: "#16a085",  # Very satisfied
        6: "#3498db",  # for nps 6-10
        7: "#2980b9",
        8: "#8e44ad",
        9: "#2c3e50",
        10: "#27ae60",
    }

    # Stacked bars (100%)
    for score in review_range:

        if show_counts:
            y = counts[score]
        else:
            y = pct[score]
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=y,
                name=str(score),
                marker_color=color_map[score],
                # hovertemplate=f"Period: %{ { 'x' } }<br>Score {score}: %{ { 'y' }:.1f}%<extra></extra>",
            ),
            secondary_y=False,
        )

    # Average line on secondary axis
    fig.add_trace(
        go.Scatter(
            x=means[period_col],
            y=means[score_col],
            mode="lines+markers+text",
            text=[f"{v:.1f}" for v in means[score_col]],
            textposition="top center",
            textfont=dict(color="black", size=10),
            line=dict(color="black", width=2),
            marker=dict(size=6, color="black"),
            name="Average",
            hovertemplate="Period: %{x}<br>Average: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Layout to mimic the screenshot
    fig.update_layout(
        barmode="stack",
        title=dict(
            text=f"Indicator evolution {score_col}"
        ),  # , x=0.02, y=0.98, font=dict(size=16)),
        legend_title_text="Score",
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        bargap=0.1,
    )

    # Left axis = percentages
    if not show_counts:
        fig.update_yaxes(title_text="%", range=[0, 100], secondary_y=False)
    # Right axis = average scale roughly 3–5.5 (like the image)
    fig.update_yaxes(title_text="Average", range=[0.0, rev_max + 0.5], secondary_y=True)

    fig.update_xaxes(title_text="Period")
    st.plotly_chart(
        fig, use_container_width=True, key=f"{score_col}-{random.randint(1,10000)}"
    )
    return


def create_heatmap(grouper, result, nps_per_type):
    """Make heatmap with NPS (1–10), other scores (1–5), and NPS (-100..100) per branche.

    Args:
        grouper (str): column name to group by (e.g. 'branche')
        result (pd.dataframe): summary dataframe with mean scores per type
        nps_per_type (pd.dataframe): dataframe with NPS per type
    Returns
        merged(pd.dataframe)
    """

    # ------- labels met review_count -------
    result["label"] = result[grouper] + " (" + result["review_count"].astype(str) + ")"

    # sorteer volgorde op NPS -100..100
    nps_per_type = nps_per_type.merge(
        result[[grouper, "label"]], on=grouper, how="left"
    )
    y_order = nps_per_type.sort_values("NPS")["label"].tolist()

    # ------- NPS (1–10) -------
    nps = (
        result.set_index("label")["mean_nps"].reindex(y_order).to_numpy().reshape(-1, 1)
    )
    heatmap_nps = go.Heatmap(
        z=nps,
        x=["NPS 1–10"],
        y=y_order,
        text=np.where(np.isfinite(nps), np.round(nps, 1).astype(object), ""),
        texttemplate="%{text}",
        zmin=1,
        zmax=10,
        colorscale=[
            [0.0, "red"],
            [0.6, "red"],
            [0.6, "orange"],
            [0.8, "orange"],
            [0.8, "green"],
            [1.0, "green"],
        ],
        showscale=False,
        hoverinfo="skip",
    )

    # ------- Scores (1–5) -------

    cols = [
        "mean_Instructions",
        "mean_Connection",
        "mean_Adjustments",
        "mean_Tangibles",
        "mean_Knowledge",
    ]
    M = result.set_index("label")[cols].reindex(y_order)
    Z = M.to_numpy()
    thr = 4.0
    pos_thr = (thr - 1) / (5 - 1)
    colors_other = [
        [0.0, "darkred"],
        [max(pos_thr - 1e-6, 0.0), "lightcoral"],
        [pos_thr, "lightgreen"],
        [1.0, "darkgreen"],
    ]
    heatmap_other = go.Heatmap(
        z=Z,
        x=M.columns.tolist(),
        y=y_order,
        text=np.where(np.isfinite(Z), np.round(Z, 1).astype(object), ""),
        texttemplate="%{text}",
        zmin=1,
        zmax=5,
        colorscale=colors_other,
        showscale=False,
        hoverinfo="skip",
    )

    # ------- NPS (-100..100) -------
    nps100 = (
        nps_per_type.set_index("label")["NPS"]
        .reindex(y_order)
        .to_numpy()
        .reshape(-1, 1)
    )
    norm = lambda v: (v + 100) / 200.0
    colors_nps100 = [
        [0.0, "red"],
        [norm(0), "red"],
        [norm(0), "orange"],
        [norm(50), "orange"],
        [norm(50), "green"],
        [1.0, "green"],
    ]
    heatmap_nps100 = go.Heatmap(
        z=nps100,
        x=["NPS -100..100"],
        y=y_order,
        text=np.where(np.isfinite(nps100), np.round(nps100, 0).astype(object), ""),
        texttemplate="%{text}",
        zmin=-100,
        zmax=100,
        colorscale=colors_nps100,
        showscale=False,
        hoverinfo="skip",
    )

    # ------- Combine -------
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.06)
    fig.add_trace(heatmap_nps, row=1, col=1)
    fig.add_trace(heatmap_other, row=1, col=2)
    fig.add_trace(heatmap_nps100, row=1, col=3)

    fig.update_layout(
        title="Client Satisfaction per branche",
        yaxis_title="branche",
        height=600,
        width=1400,
    )

    st.plotly_chart(fig, use_container_width=True)

    # to show scatterplots

    # # --- Merge NPS table with mean scores ---
    merged = nps_per_type.merge(result, on=grouper, how="left")
    return merged


def generate_data():
    """We generate data. 100 imaginary staffmembers.
     We generate also 200 survey result. Answers skewed towards higher numbers

    Returns:
        _type_: _description_
    """
    # Parameters
    rows = 200
    dates = pd.date_range("2025-04-01", "2025-10-01", freq="D")
    staffmembernrs = np.arange(1, 101)

    # Weighted distributions
    # More 5's than 1's
    ratings = [1, 2, 3, 4, 5]
    rating_weights = [0.02, 0.08, 0.20, 0.30, 0.40]
    # rating_weights_tangibles = [0.15, 0.15, 0.20, 0.20, 0.30]

    # NPS: skewed towards higher values
    nps_values = list(range(0, 11))
    nps_weights = [0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.04, 0.07, 0.11, 0.30, 0.37]

    np.random.seed(42)
    data = {
        "datum": np.random.choice(dates, rows),
        "staffmembercode": np.random.randint(100, 1000, rows),  # random staffmembercode
        "staffmembernr": np.random.choice(staffmembernrs, rows),
        # "branche_": ["" for _ in range(rows)],  # empty strings
        # "bookingnr": np.random.randint(100000, 999999, rows),
        "nps": np.random.choice(nps_values, rows, p=nps_weights),
        "Aftersales": np.random.choice(ratings, rows, p=rating_weights),
        "Instructions": np.random.choice(ratings, rows, p=rating_weights),
        "Connection": np.random.choice(ratings, rows, p=rating_weights),
        "Adjustments": np.random.choice(ratings, rows, p=rating_weights),
        "Tangibles": np.random.choice(ratings, rows, p=rating_weights),
        "Knowledge": np.random.choice(ratings, rows, p=rating_weights),
    }

    df_skewed = pd.DataFrame(data)

    return df_skewed


def generate_staffmembers():
    """We generate 100 imaginary staffmembers working at  10 branches (10 of each)

    Returns:
        df: dataframe with staffmembernr and branche
    """
    # Generate acconr 1-100
    staffmembernrs = list(range(1, 101))

    # Define accotypes (10 of each)
    # types = ["alfa", "bravo", "charlie", "delta", "echo",
    # "foxtrot", "golf", "hotel", "india", "juliet"]
    types = [
        "Asana",
        "Bandha",
        "Chakra",
        "Dhyana",
        "Eka",
        "Flow",
        "Guru",
        "Hatha",
        "Ishvara",
        "Japa",
    ]

    # Asana – posture or pose
    # Bandha – body lock to control energy
    # Chakra – energy center in the body
    # Dhyana – meditation
    # Eka – “one” in Sanskrit, used in poses (e.g. Eka Pada)
    # Flow – often used in modern yoga for vinyasa sequences
    # Guru – teacher or master
    # Hatha – branch of yoga focused on physical practice
    # Ishvara – higher self or divine principle
    # Japa – repetitive chanting of a mantra
    branches = [f"{t}_street" for t in types for _ in range(10)]

    # Create DataFrame
    df_staffmember = pd.DataFrame(
        {"staffmembernr": staffmembernrs, "branche": branches}
    )

    return df_staffmember


def main():
    st.title("Analyze Reviews Dummy Data")
    st.info("https://rene-smit.com/analysis-of-student-satisfaction-at-yepyoga/")
    period = st.selectbox("Period", ["week", "month", "year"])  # unchanged

    df_ = generate_data()
    df_["country"] = "Zulu"
    df_branches = generate_staffmembers()
    df__ = pd.merge(left=df_, right=df_branches, on=["staffmembernr"])  # unchanged
    # -----------------------------
    # NEW: multiselect for branche
    # -----------------------------
    branche_options = sorted(df__["branche"].dropna().unique())
    selected_branches = st.multiselect(
        "Select branches", branche_options, default=branche_options
    )

    df__ = df__[df__["branche"].isin(selected_branches)]
    # Convert datum to datetime
    df__["datum"] = pd.to_datetime(df__["datum"], dayfirst=True, errors="coerce")
    if period == "week":
        p = "W"
    elif period == "month":
        p = "M"
    elif period == "year":
        p = "Y"

    else:
        st.error("Error in period")
        st.stop()

    # Extract year-period
    df__["year_period"] = df__["datum"].dt.to_period(p).astype(str)
    df_Aftersales = df__[df__["Aftersales"].notna()].copy()

    df = df__[df__["nps"].notna()].copy()
    # df = df[df["nps"] != 0 ] #in the dataset, people could also give a 0 for nps. If I filter these out, the results
    # are not the same as in the report
    if len(selected_branches) == 0:
        st.error("Please select at least one branche.")
        st.stop()

    if len(df) == 0:
        st.error("No data for selected branches.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of reviews", len(df))
        st.metric("Number of reviews Aftersales", len(df_Aftersales))

    with col2:
        nps_overall = compute_nps(df, ["country"])

        st.metric("NPS overall", nps_overall["NPS"].values[0])
        nps_per_type = compute_nps(df, ["branche"])

    with col3:
        st.metric("Mean NPS-rating", round(df["nps"].mean(), 1))
    # st.write(df)

    result = summarize_ratings_all_columns(df, "branche")

    merged = create_heatmap("branche", result, nps_per_type)
    tab_titles = [
        "nps",
        "Instructions",
        "Connection",
        "Adjustments",
        "Tangibles",
        "Knowledge",
    ]
    tabs = st.tabs(
        tab_titles + ["NPS/per.", "aftersales.sc./per", "OLS/corr.", "Reviews/per."]
    )
    for i, what in enumerate(tab_titles):
        with tabs[i]:
            # Make sure rating columns are numeric
            df[what] = pd.to_numeric(df[what], errors="coerce")
            plot_indicator_evolution(df, period_col="year_period", score_col=what)

            for grouper in ["country", "branche"]:
                result = summarize_ratings(df, grouper, what)

                st.subheader(f"Review Analysis : {grouper} - {what}")
                st.write(result.sort_values(f"mean_{what}", ascending=False))
                if (grouper == "branche") & (what == "nps"):
                    plot_nps_per_branche(result)
                # Mean <what> per period per group
                period_avg = (
                    df.groupby(["year_period", grouper])[what].mean().reset_index()
                )

                # Plot with Plotly
                fig = px.line(
                    period_avg,
                    x="year_period",
                    y=what,
                    color=grouper,
                    markers=True,
                    title=f"Average {what} Score per period by {grouper}",
                )

                fig.update_layout(
                    xaxis_title="period",
                    yaxis_title=f"Average {what} Score",
                    legend_title=grouper,
                )
                st.plotly_chart(fig)
                st.write(
                    "Double click on an branche in the legenda to see a correct line"
                )
                # Usage inside your Streamlit tab where you already have df with year_period & Aftersales:

            if what != "nps":
                make_scatterplot(df, "nps", what)
                make_scatterplot(merged, "NPS", f"mean_{what}")

    with tabs[6]:
        nps_per_period(df)
    with tabs[7]:
        Aftersales_score_per_period(df_Aftersales)
    with tabs[8]:
        ols_corr(df)
    with tabs[9]:
        reviews_per_period(df)


if __name__ == "__main__":
    main()
