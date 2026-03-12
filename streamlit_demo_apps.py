
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

import yfinance as yf
import random

st.set_page_config(
    page_title="Streamlit Dashboard App Templates",
    page_icon=":material/business:",
    layout="wide",
)
# =============================================================================
# Page Header Component
# =============================================================================
def render_page_header(title: str):
    """Render page header with title and reset button."""
    uniquekey = random.randint(0,1000000)
    with st.container(
        horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"
    ):
        st.markdown(title)
        if st.button(":material/restart_alt: Reset", type="tertiary", key=uniquekey):
            st.session_state.clear()
            st.rerun()

# =============================================================================
# Chart Utilities
# =============================================================================
def filter_by_time_range(df: pd.DataFrame, x_col: str, time_range: str) -> pd.DataFrame:
    """Filter dataframe by time range."""
    if time_range == "All" or df.empty:
        return df

    df = df.copy()
    df[x_col] = pd.to_datetime(df[x_col])
    max_date = df[x_col].max()

    if time_range == "1M":
        min_date = max_date - timedelta(days=30)
    elif time_range == "6M":
        min_date = max_date - timedelta(days=180)
    elif time_range == "1Y":
        min_date = max_date - timedelta(days=365)
    elif time_range == "QTD":
        quarter_month = ((max_date.month - 1) // 3) * 3 + 1
        min_date = pd.Timestamp(date(max_date.year, quarter_month, 1))
    elif time_range == "YTD":
        min_date = pd.Timestamp(date(max_date.year, 1, 1))
    else:
        return df

    return df[df[x_col] >= min_date]


def dashboard_companies():
    """
    Company Analytics Dashboard Template

    A company leaderboard dashboard demonstrating:
    - Interactive dataframe with sparkline columns
    - Segmented control for ranking (top spenders, gainers, shrinkers)
    - Multi-select pills for account type filtering
    - Time window filtering
    - Growth score calculation
    - Dialog popup for company details

    This template uses synthetic data. Replace generate_company_data()
    with your actual data source (e.g., Snowflake queries, CRM APIs, etc.)
    """

  
    # =============================================================================
    # Synthetic Data Generation (Replace with your data source)
    # =============================================================================

    COMPANY_NAMES = [
        "Acme Corp", "TechFlow Inc", "DataDriven Co", "CloudFirst Ltd", 
        "InnovateTech", "ScaleUp Systems", "PrimeData Inc", "FutureStack",
        "ByteWise Corp", "StreamLine Co", "Quantum Labs", "NexGen Solutions",
        "AlphaMetrics", "BetaAnalytics", "GammaInsights", "DeltaData",
        "OmegaTech", "SigmaSoft", "ThetaCloud", "ZetaDigital",
    ]

    ACCOUNT_TYPES = ["Enterprise", "Growth", "Startup", "Trial", "Internal"]
    REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
    SEGMENTS = ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing"]


    @st.cache_data(ttl=3600)
    def generate_company_data(days: int = 90) -> pd.DataFrame:
        """Generate synthetic company usage data.
        
        Replace this function with your actual data source.
        """
        np.random.seed(42)
        
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        
        records = []
        
        for company in COMPANY_NAMES:
            # Assign static attributes
            account_type = np.random.choice(ACCOUNT_TYPES, p=[0.3, 0.25, 0.2, 0.15, 0.1])
            region = np.random.choice(REGIONS)
            segment = np.random.choice(SEGMENTS)
            
            # Generate usage pattern
            base_usage = np.random.randint(100, 10000)
            growth = np.random.uniform(-0.005, 0.01)  # Some companies shrink
            
            for i, dt in enumerate(dates):
                # Base trend
                trend = base_usage * (1 + growth) ** i
                
                # Weekly seasonality
                if dt.dayofweek >= 5:
                    trend *= 0.3
                
                # Random noise
                daily_credits = max(0, trend * np.random.uniform(0.7, 1.3))
                
                records.append({
                    "company_name": company,
                    "date": dt,
                    "daily_credits": daily_credits,
                    "account_type": account_type,
                    "region": region,
                    "segment": segment,
                })
        
        return pd.DataFrame(records)


    @st.cache_data(ttl=3600)
    def load_company_data() -> pd.DataFrame:
        """Load all company data."""
        return generate_company_data(days=90)


    def aggregate_companies(
        df: pd.DataFrame,
        days: int | None = None,
        account_types: list[str] | None = None,
        sort_by: str = "total_credits",
    ) -> pd.DataFrame:
        """Filter and aggregate company data."""
        result = df.copy()
        
        # Filter by time window
        if days:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
            result = result[result["date"] >= cutoff]
        
        # Filter by account type
        if account_types:
            result = result[result["account_type"].isin(account_types)]
        
        if result.empty:
            return pd.DataFrame()
        
        # Aggregate to company level
        agg = result.groupby("company_name").agg(
            total_credits=("daily_credits", "sum"),
            active_days=("date", "nunique"),
            account_type=("account_type", "first"),
            region=("region", "first"),
            segment=("segment", "first"),
        ).reset_index()
        
        # Calculate daily average
        agg["daily_avg"] = agg["total_credits"] / agg["active_days"]
        
        # Build sparkline data (list of daily values)
        sparklines = (
            result.groupby("company_name")
            .apply(lambda x: x.sort_values("date")["daily_credits"].tolist())
            .reset_index()
        )
        sparklines.columns = ["company_name", "usage_trend"]
        agg = agg.merge(sparklines, on="company_name")
        
        # Calculate growth score (second half vs first half)
        def calc_growth(trend):
            if not trend or len(trend) < 2:
                return 0
            mid = len(trend) // 2
            first_half = sum(trend[:mid]) if mid > 0 else 0
            second_half = sum(trend[mid:])
            return second_half - first_half
        
        agg["growth_score"] = agg["usage_trend"].apply(calc_growth)
        
        # Sort
        if sort_by == "growth_asc":
            agg = agg.sort_values("growth_score", ascending=True)
        elif sort_by == "growth_desc":
            agg = agg.sort_values("growth_score", ascending=False)
        else:
            agg = agg.sort_values("total_credits", ascending=False)
        
        return agg


    def render_company_dialog(company_name: str, company_row: pd.Series, df: pd.DataFrame):
        """Render company details inside a dialog."""
        company_data = df[df["company_name"] == company_name].sort_values("date")
        
        if company_data.empty:
            st.warning("No data available for this company.")
            return
        
        # Company info badges - extract from list format back to single value
        account_type = company_row["account_type"][0] if company_row["account_type"] else "Unknown"
        region = company_row["region"][0] if company_row["region"] else "Unknown"
        segment = company_row["segment"][0] if company_row["segment"] else "Unknown"
        total_credits = company_row["total_credits"]
        
        st.markdown(
            f":blue-badge[{account_type}] "
            f":violet-badge[{region}] "
            f":orange-badge[{segment}] "
            f":green-badge[{total_credits:,.0f} credits]"
        )
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Credits", f"{total_credits:,.0f}")
        with col2:
            st.metric("Active Days", f"{company_row['active_days']:,}")
        with col3:
            growth = company_row["growth_score"]
            st.metric("Growth Score", f"{growth:+,.0f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.markdown("**Daily usage**")
                st.line_chart(company_data, x="date", y="daily_credits", height=250)
        
        with col2:
            with st.container(border=True):
                st.markdown("**Cumulative usage**")
                chart_data = company_data.copy()
                chart_data["cumulative"] = chart_data["daily_credits"].cumsum()
                st.area_chart(chart_data, x="date", y="cumulative", height=250)


    # =============================================================================
    # Page Layout
    # =============================================================================

    # Load data
    all_data = load_company_data()

    st.markdown("# :material/business: Company Analytics")
    st.caption("Track company adoption - usage, growth trends, and account details.")

    # Filters
    with st.container(border=True):
        st.markdown("**Filters**")
        
        # Company selection mode
        sort_mode = st.segmented_control(
            "Sort by",
            options=[
                "All companies",
                ":material/military_tech: Top spenders",
                ":material/trending_down: Top shrinkers",
                ":material/trending_up: Top gainers",
            ],
            default="All companies",
        )
        
        # Time window
        timeframe_options = {
            "All time": None,
            "Last 28 days": 28,
            "Last 7 days": 7,
        }
        timeframe = st.segmented_control(
            "Time window",
            options=list(timeframe_options.keys()),
            default="Last 28 days",
        )
        days_filter = timeframe_options.get(timeframe)
        
        # Account types
        account_types = st.pills(
            "Account types",
            options=ACCOUNT_TYPES,
            default=["Enterprise", "Growth", "Startup"],
            selection_mode="multi",
        )

    # Determine sort order
    if "Top shrinkers" in (sort_mode or ""):
        sort_by = "growth_asc"
    elif "Top gainers" in (sort_mode or ""):
        sort_by = "growth_desc"
    else:
        sort_by = "total_credits"

    # Get filtered data
    leaderboard = aggregate_companies(
        all_data,
        days=days_filter,
        account_types=account_types,
        sort_by=sort_by,
    )

    if leaderboard.empty:
        st.warning("No company data found for the selected filters.")
        st.stop()


    def _to_list(val):
        """Convert a single value to a list for MultiselectColumn display."""
        return [val] if pd.notna(val) else []


    # Convert columns to lists for MultiselectColumn display (shows nice colored chips)
    for col in ["account_type", "region", "segment"]:
        leaderboard[col] = leaderboard[col].apply(_to_list)

    # Companies dataframe
    with st.container(border=True):
        timeframe_text = timeframe.lower() if timeframe != "All time" else "all time"
        st.markdown(f"**Companies — {timeframe_text}**")
        
        # Selection dataframe with cell-click support
        selection = st.dataframe(
            leaderboard,
            column_config={
                "company_name": st.column_config.TextColumn(
                    "Company (👋 click to view details)",
                    width="medium",
                ),
                "account_type": st.column_config.MultiselectColumn(
                    "Type",
                    options=ACCOUNT_TYPES,
                    color="auto",
                    width="small",
                ),
                "total_credits": st.column_config.NumberColumn(
                    "Credits",
                    format="%.0f",
                ),
                "growth_score": st.column_config.NumberColumn(
                    "Growth",
                    format="%+.0f",
                    help="Credit change: second half vs first half of period",
                ),
                "usage_trend": st.column_config.LineChartColumn(
                    "Trend",
                    width="medium",
                ),
                "daily_avg": st.column_config.NumberColumn(
                    "Daily Avg",
                    format="%.1f",
                ),
                "active_days": st.column_config.NumberColumn(
                    "Active Days",
                    format="%d",
                ),
                "region": st.column_config.MultiselectColumn(
                    "Region",
                    options=REGIONS,
                    color="auto",
                ),
                "segment": st.column_config.MultiselectColumn(
                    "Segment",
                    options=SEGMENTS,
                    color="auto",
                ),
            },
            column_order=[
                "company_name", "account_type", "total_credits", "growth_score",
                "usage_trend", "daily_avg", "region", "segment",
            ],
            hide_index=True,
            on_select="rerun",
            selection_mode="single-cell",
            key="company_leaderboard",
        )

    # Company drill-down via dialog when Company column cell is clicked
    if selection.selection.cells:
        cell = selection.selection.cells[0]  # tuple: (row_index, column_name)
        row_idx, col_name = cell
        # Check if the clicked cell is in the company_name column
        if col_name == "company_name":
            selected_company = leaderboard.iloc[row_idx]["company_name"]
            company_row = leaderboard.iloc[row_idx]

            @st.dialog(f"{selected_company}", width="large")
            def show_company_dialog():
                render_company_dialog(
                    selected_company,
                    company_row=company_row,
                    df=all_data,
                )

            show_company_dialog()

def dashboard_metrics():
    """
    Metrics Dashboard Template

    A comprehensive metrics dashboard demonstrating:
    - Time series visualization with Altair (line, area, bar, point charts)
    - Metric cards with chart/table toggle and popover filters
    - Time range filtering (1M, 6M, 1Y, QTD, YTD, All)
    - Line options (Daily, 7-day MA)

    This template uses synthetic data. Replace the generate_*_data() functions
    with your own data sources (e.g., Snowflake queries, APIs, etc.)
    """




    # =============================================================================
    # Constants
    # =============================================================================

    TIME_RANGES = ["1M", "6M", "1Y", "QTD", "YTD", "All"]
    CHART_HEIGHT = 300


    # =============================================================================
    # Synthetic Data Generation (Replace with your data source)
    # =============================================================================


    def generate_metric_data(
        metric_name: str,
        start_date: date,
        end_date: date,
        base_value: float = 1000,
        growth_rate: float = 0.001,
        noise_factor: float = 0.1,
    ) -> pd.DataFrame:
        """Generate synthetic time series data for a metric.
        
        Replace this function with your actual data source, e.g.:
        - Snowflake query
        - API call
        - Database query
        """
        np.random.seed(hash(metric_name) % 2**32)
        
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        n_days = len(dates)
        
        # Base trend with growth
        trend = base_value * (1 + growth_rate) ** np.arange(n_days)
        
        # Add weekly seasonality (lower on weekends)
        day_of_week = dates.dayofweek
        seasonality = np.where(day_of_week >= 5, 0.7, 1.0)
        trend = trend * seasonality
        
        # Add noise
        noise = np.random.normal(1, noise_factor, n_days)
        values = trend * noise
        
        # Calculate rolling averages
        df = pd.DataFrame({
            "ds": dates,
            "daily_value": values,
        })
        df["value_7d_ma"] = df["daily_value"].rolling(7, min_periods=1).mean()
        
        return df


    @st.cache_data(ttl=3600)
    def load_all_metrics() -> dict[str, pd.DataFrame]:
        """Load all metrics data. Replace with your data loading logic."""
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=730)  # 2 years of data
        
        return {
            "users": generate_metric_data("users", start_date, end_date, base_value=5000, growth_rate=0.002),
            "sessions": generate_metric_data("sessions", start_date, end_date, base_value=15000, growth_rate=0.003),
            "revenue": generate_metric_data("revenue", start_date, end_date, base_value=50000, growth_rate=0.001),
            "conversions": generate_metric_data("conversions", start_date, end_date, base_value=500, growth_rate=0.0015),
        }


    
    def render_line_chart(
        df: pd.DataFrame,
        x_col: str,
        y_cols: list[str],
        labels: list[str],
        height: int = CHART_HEIGHT,
    ) -> alt.Chart:
        """Render a multi-line chart."""
        # Melt for Altair
        melted = df.melt(
            id_vars=[x_col],
            value_vars=y_cols,
            var_name="series",
            value_name="value",
        )
        
        # Map to labels
        label_map = dict(zip(y_cols, labels))
        melted["series"] = melted["series"].map(label_map)
        
        chart = (
            alt.Chart(melted)
            .mark_line()
            .encode(
                x=alt.X(f"{x_col}:T", title=None),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom")),
                strokeDash=alt.condition(
                    alt.datum.series == "7-day MA",
                    alt.value([5, 5]),
                    alt.value([0]),
                ),
                tooltip=[
                    alt.Tooltip(f"{x_col}:T", title="Date", format="%Y-%m-%d"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("value:Q", title="Value", format=",.0f"),
                ],
            )
            .properties(height=height)
        )
        
        return chart


    def render_area_chart(
        df: pd.DataFrame,
        x_col: str,
        y_cols: list[str],
        labels: list[str],
        height: int = CHART_HEIGHT,
    ) -> alt.Chart:
        """Render a stacked area chart."""
        melted = df.melt(
            id_vars=[x_col],
            value_vars=y_cols,
            var_name="series",
            value_name="value",
        )
        label_map = dict(zip(y_cols, labels))
        melted["series"] = melted["series"].map(label_map)

        chart = (
            alt.Chart(melted)
            .mark_area(opacity=0.6, line=True)
            .encode(
                x=alt.X(f"{x_col}:T", title=None),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom")),
                tooltip=[
                    alt.Tooltip(f"{x_col}:T", title="Date", format="%Y-%m-%d"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("value:Q", title="Value", format=",.0f"),
                ],
            )
            .properties(height=height)
        )
        return chart


    def render_bar_chart(
        df: pd.DataFrame,
        x_col: str,
        y_cols: list[str],
        labels: list[str],
        height: int = CHART_HEIGHT,
    ) -> alt.Chart:
        """Render a bar chart (weekly aggregation for readability)."""
        df = df.copy()
        df[x_col] = pd.to_datetime(df[x_col])
        df["week"] = df[x_col].dt.to_period("W").dt.start_time

        # Aggregate by week
        agg_df = df.groupby("week")[y_cols].mean().reset_index()

        melted = agg_df.melt(
            id_vars=["week"],
            value_vars=y_cols,
            var_name="series",
            value_name="value",
        )
        label_map = dict(zip(y_cols, labels))
        melted["series"] = melted["series"].map(label_map)

        chart = (
            alt.Chart(melted)
            .mark_bar(opacity=0.8)
            .encode(
                x=alt.X("week:T", title=None),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom")),
                xOffset="series:N",
                tooltip=[
                    alt.Tooltip("week:T", title="Week", format="%Y-%m-%d"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("value:Q", title="Value", format=",.0f"),
                ],
            )
            .properties(height=height)
        )
        return chart


    def render_point_chart(
        df: pd.DataFrame,
        x_col: str,
        y_cols: list[str],
        labels: list[str],
        height: int = CHART_HEIGHT,
    ) -> alt.Chart:
        """Render a scatter/point chart with trend line."""
        melted = df.melt(
            id_vars=[x_col],
            value_vars=y_cols,
            var_name="series",
            value_name="value",
        )
        label_map = dict(zip(y_cols, labels))
        melted["series"] = melted["series"].map(label_map)

        points = (
            alt.Chart(melted)
            .mark_point(opacity=0.5, size=20)
            .encode(
                x=alt.X(f"{x_col}:T", title=None),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom")),
                tooltip=[
                    alt.Tooltip(f"{x_col}:T", title="Date", format="%Y-%m-%d"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("value:Q", title="Value", format=",.0f"),
                ],
            )
        )

        # Add trend line for 7-day MA only
        trend = (
            alt.Chart(melted[melted["series"] == "7-day MA"])
            .mark_line(strokeDash=[5, 5], strokeWidth=2)
            .encode(
                x=alt.X(f"{x_col}:T"),
                y=alt.Y("value:Q"),
                color=alt.Color("series:N"),
            )
        )

        return (points + trend).properties(height=height)


    # =============================================================================
    # Metric Card Component
    # =============================================================================


    def metric_card(
        title: str,
        df: pd.DataFrame,
        key_prefix: str,
        chart_type: str = "line",
    ):
        """Display a metric card with chart/table toggle and popover filters.

        Args:
            title: Card title
            df: DataFrame with ds, daily_value, value_7d_ma columns
            key_prefix: Unique prefix for widget keys
            chart_type: One of "line", "area", "bar", "point"
        """
        chart_renderers = {
            "line": render_line_chart,
            "area": render_area_chart,
            "bar": render_bar_chart,
            "point": render_point_chart,
        }
        render_chart = chart_renderers.get(chart_type, render_line_chart)

        with st.container(border=True):
            # Header row with title, view toggle, and filters
            with st.container(
                horizontal=True,
                horizontal_alignment="distribute",
                vertical_alignment="center",
            ):
                st.markdown(f"**{title}**")
                
                view_mode = st.segmented_control(
                    "View",
                    options=[":material/show_chart:", ":material/table:"],
                    default=":material/show_chart:",
                    key=f"{key_prefix}_view",
                    label_visibility="collapsed",
                )
                
                with st.popover("Filters", type="tertiary"):
                    line_options = st.pills(
                        "Lines",
                        options=["Daily", "7-day MA"],
                        default=["Daily", "7-day MA"],
                        selection_mode="multi",
                        key=f"{key_prefix}_lines",
                    )
                    time_range = st.segmented_control(
                        "Time range",
                        options=TIME_RANGES,
                        default="All",
                        key=f"{key_prefix}_time",
                    )
            
            # Apply filters
            line_options = line_options or ["7-day MA"]
            filtered_df = filter_by_time_range(df, "ds", time_range)
            
            # Determine which columns to show
            y_cols = []
            labels = []
            if "Daily" in line_options:
                y_cols.append("daily_value")
                labels.append("Daily")
            if "7-day MA" in line_options:
                y_cols.append("value_7d_ma")
                labels.append("7-day MA")
            
            # Render view
            if "table" in (view_mode or ""):
                st.dataframe(
                    filtered_df,
                    height=CHART_HEIGHT,
                    hide_index=True,
                )
            else:
                if y_cols:
                    st.altair_chart(
                        render_chart(filtered_df, "ds", y_cols, labels),
                    )
                else:
                    st.info("Select at least one line option.")




    # =============================================================================
    # Page Layout
    # =============================================================================

    # Load data (cached)
    metrics_data = load_all_metrics()

    # Page header
    render_page_header("# :material/monitoring: Metrics Dashboard")

    # Row 1: Users and Sessions
    row1 = st.columns(2)

    with row1[0]:
        metric_card("Active Users", metrics_data["users"], "users", chart_type="line")

    with row1[1]:
        metric_card("Sessions", metrics_data["sessions"], "sessions", chart_type="area")

    # Row 2: Revenue and Conversions
    row2 = st.columns(2)

    with row2[0]:
        metric_card("Revenue", metrics_data["revenue"], "revenue", chart_type="bar")

    with row2[1]:
        metric_card("Conversions", metrics_data["conversions"], "conversions", chart_type="point")


def dashboard_compute():
    """
    Compute/Resource Dashboard Template

    A resource consumption dashboard demonstrating:
    - Multiple metric cards in a grid layout
    - @st.fragment for independent widget updates
    - Popover filters for each metric card
    - Chart/table view toggle
    - Time range filtering (1M, 6M, 1Y, QTD, YTD, All)
    - Percentage normalization toggle
    - Multiple breakdown dimensions

    This template uses synthetic data. Replace generate_*_data()
    with your actual data source (e.g., Snowflake queries, cloud APIs, etc.)
    """

    # =============================================================================
    # Constants
    # =============================================================================

    TIME_RANGES = ["1M", "6M", "1Y", "QTD", "YTD", "All"]
    ACCOUNT_TYPES = ["Paying", "Trial", "Internal"]
    INSTANCE_TYPES = ["Standard", "High Memory", "High CPU", "GPU"]
    REGIONS = ["us-west-2", "us-east-1", "eu-west-1", "ap-northeast-1"]
    CHART_HEIGHT = 350


    # =============================================================================
    # Synthetic Data Generation
    # =============================================================================


    def generate_time_series(
        categories: list[str],
        category_name: str,
        start_date: date,
        end_date: date,
        base_values: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic time series data by category."""
        np.random.seed(hash(category_name) % 2**32)
        
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        records = []
        
        for category in categories:
            base = base_values.get(category, 1000) if base_values else np.random.randint(500, 5000)
            growth = np.random.uniform(0.001, 0.005)
            
            for i, dt in enumerate(dates):
                trend = base * (1 + growth) ** i
                if dt.dayofweek >= 5:
                    trend *= 0.4
                
                daily = max(0, trend * np.random.uniform(0.8, 1.2))
                
                records.append({
                    "ds": dt,
                    category_name: category,
                    "daily_credits": daily,
                })
        
        df = pd.DataFrame(records)
        
        # Add 7-day moving average
        df["credits_7d_ma"] = (
            df.groupby(category_name)["daily_credits"]
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )
        
        return df


    @st.cache_data(ttl=3600)
    def load_account_type_data() -> pd.DataFrame:
        """Load credits by account type."""
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=730)  # 2 years
        return generate_time_series(
            ACCOUNT_TYPES, "account_type", start_date, end_date,
            base_values={"Paying": 8000, "Trial": 2000, "Internal": 1000},
        )


    @st.cache_data(ttl=3600)
    def load_instance_type_data() -> pd.DataFrame:
        """Load credits by instance type."""
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=730)
        return generate_time_series(
            INSTANCE_TYPES, "instance_type", start_date, end_date,
            base_values={"Standard": 5000, "High Memory": 3000, "High CPU": 2000, "GPU": 1500},
        )


    @st.cache_data(ttl=3600)
    def load_region_data() -> pd.DataFrame:
        """Load credits by region."""
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=730)
        return generate_time_series(
            REGIONS, "region", start_date, end_date,
            base_values={"us-west-2": 4000, "us-east-1": 3500, "eu-west-1": 2500, "ap-northeast-1": 1500},
        )


    


    def create_line_chart(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str,
        height: int,
        show_percent: bool = False,
    ) -> alt.Chart:
        """Create a line chart."""
        y_format = ".1%" if show_percent else ",.0f"
        
        return (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X(f"{x_col}:T", title=None),
                y=alt.Y(f"{y_col}:Q", title="Credits", axis=alt.Axis(format=y_format)),
                color=alt.Color(f"{color_col}:N", legend=alt.Legend(orient="bottom")),
                tooltip=[
                    alt.Tooltip(f"{x_col}:T", title="Date", format="%Y-%m-%d"),
                    alt.Tooltip(f"{color_col}:N", title=color_col.replace("_", " ").title()),
                    alt.Tooltip(f"{y_col}:Q", title="Credits", format=y_format),
                ],
            )
            .properties(height=height)
            .interactive()
        )


    def create_bar_chart(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str,
        height: int,
        show_percent: bool = False,
    ) -> alt.Chart:
        """Create a stacked bar chart."""
        y_format = ".1%" if show_percent else ",.0f"
        
        return (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(f"{x_col}:T", title=None),
                y=alt.Y(
                    f"{y_col}:Q",
                    title="Credits",
                    stack="normalize" if show_percent else True,
                    axis=alt.Axis(format=y_format),
                ),
                color=alt.Color(f"{color_col}:N", legend=alt.Legend(orient="bottom")),
                tooltip=[
                    alt.Tooltip(f"{x_col}:T", title="Date", format="%Y-%m-%d"),
                    alt.Tooltip(f"{color_col}:N"),
                    alt.Tooltip(f"{y_col}:Q", format=",.0f"),
                ],
            )
            .properties(height=height)
        )


  
    # =============================================================================
    # Metric Card Components (using @st.fragment)
    # =============================================================================


    @st.fragment
    def account_type_metric():
        """Account type metric card with independent state."""
        data = load_account_type_data()
        
        with st.container(border=True):
            with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
                st.markdown("**Credits by account type**")
                
                view_mode = st.segmented_control(
                    "View",
                    options=[":material/show_chart:", ":material/table:"],
                    default=":material/show_chart:",
                    key="acct_view",
                    label_visibility="collapsed",
                )
                
                with st.popover("Filters", type="tertiary"):
                    selected_types = st.pills(
                        "Account types",
                        options=ACCOUNT_TYPES,
                        default=["Paying"],
                        selection_mode="multi",
                        key="acct_types",
                    )
                    line_options = st.pills(
                        "Lines",
                        options=["Daily", "7-day MA"],
                        default=["7-day MA"],
                        selection_mode="multi",
                        key="acct_lines",
                    )
                    chart_type = st.segmented_control(
                        "Chart type",
                        options=[":material/show_chart: Line", ":material/bar_chart: Bar"],
                        default=":material/show_chart: Line",
                        key="acct_chart",
                    )
                    show_percent = st.toggle(
                        "Show %", value=False, key="acct_pct",
                        disabled="Line" in (chart_type or ""),
                    )
                    time_range = st.segmented_control(
                        "Time range",
                        options=TIME_RANGES,
                        default="All",
                        key="acct_time",
                    )
            
            # Filter data
            selected_types = selected_types or ["Paying"]
            line_options = line_options or ["7-day MA"]
            filtered = data[data["account_type"].isin(selected_types)]
            filtered = filter_by_time_range(filtered, "ds", time_range)
            
            # Determine y column
            y_col = "credits_7d_ma" if "7-day MA" in line_options else "daily_credits"
            
            if "table" in (view_mode or ""):
                st.dataframe(filtered, height=CHART_HEIGHT, hide_index=True)
            else:
                if "Bar" in (chart_type or ""):
                    st.altair_chart(
                        create_bar_chart(filtered, "ds", y_col, "account_type", CHART_HEIGHT, show_percent),
                    )
                else:
                    st.altair_chart(
                        create_line_chart(filtered, "ds", y_col, "account_type", CHART_HEIGHT),
                    )


    @st.fragment
    def instance_type_metric():
        """Instance type metric card with independent state."""
        data = load_instance_type_data()
        
        with st.container(border=True):
            with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
                st.markdown("**Credits by instance type**")
                
                view_mode = st.segmented_control(
                    "View",
                    options=[":material/show_chart:", ":material/table:"],
                    default=":material/show_chart:",
                    key="inst_view",
                    label_visibility="collapsed",
                )
                
                with st.popover("Filters", type="tertiary"):
                    selected_types = st.pills(
                        "Instance types",
                        options=INSTANCE_TYPES,
                        default=INSTANCE_TYPES,
                        selection_mode="multi",
                        key="inst_types",
                    )
                    line_options = st.pills(
                        "Lines",
                        options=["Daily", "7-day MA"],
                        default=["7-day MA"],
                        selection_mode="multi",
                        key="inst_lines",
                    )
                    chart_type = st.segmented_control(
                        "Chart type",
                        options=[":material/show_chart: Line", ":material/bar_chart: Bar"],
                        default=":material/show_chart: Line",
                        key="inst_chart",
                    )
                    show_percent = st.toggle(
                        "Show %", value=False, key="inst_pct",
                        disabled="Line" in (chart_type or ""),
                    )
                    time_range = st.segmented_control(
                        "Time range",
                        options=TIME_RANGES,
                        default="All",
                        key="inst_time",
                    )
            
            # Filter data
            selected_types = selected_types or INSTANCE_TYPES
            line_options = line_options or ["7-day MA"]
            filtered = data[data["instance_type"].isin(selected_types)]
            filtered = filter_by_time_range(filtered, "ds", time_range)
            
            y_col = "credits_7d_ma" if "7-day MA" in line_options else "daily_credits"
            
            if "table" in (view_mode or ""):
                st.dataframe(filtered, height=CHART_HEIGHT, hide_index=True)
            else:
                if "Bar" in (chart_type or ""):
                    st.altair_chart(
                        create_bar_chart(filtered, "ds", y_col, "instance_type", CHART_HEIGHT, show_percent),
                    )
                else:
                    st.altair_chart(
                        create_line_chart(filtered, "ds", y_col, "instance_type", CHART_HEIGHT),
                    )


    @st.fragment
    def region_metric():
        """Region metric card with independent state."""
        data = load_region_data()
        
        with st.container(border=True):
            with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
                st.markdown("**Credits by region**")
                
                view_mode = st.segmented_control(
                    "View",
                    options=[":material/show_chart:", ":material/table:"],
                    default=":material/show_chart:",
                    key="region_view",
                    label_visibility="collapsed",
                )
                
                with st.popover("Filters", type="tertiary"):
                    selected_regions = st.pills(
                        "Regions",
                        options=REGIONS,
                        default=REGIONS,
                        selection_mode="multi",
                        key="region_select",
                    )
                    line_options = st.pills(
                        "Lines",
                        options=["Daily", "7-day MA"],
                        default=["7-day MA"],
                        selection_mode="multi",
                        key="region_lines",
                    )
                    chart_type = st.segmented_control(
                        "Chart type",
                        options=[":material/show_chart: Line", ":material/bar_chart: Bar"],
                        default=":material/bar_chart: Bar",
                        key="region_chart",
                    )
                    show_percent = st.toggle(
                        "Show %", value=False, key="region_pct",
                        disabled="Line" in (chart_type or ""),
                    )
                    time_range = st.segmented_control(
                        "Time range",
                        options=TIME_RANGES,
                        default="All",
                        key="region_time",
                    )
            
            # Filter data
            selected_regions = selected_regions or REGIONS
            line_options = line_options or ["7-day MA"]
            filtered = data[data["region"].isin(selected_regions)]
            filtered = filter_by_time_range(filtered, "ds", time_range)
            
            y_col = "credits_7d_ma" if "7-day MA" in line_options else "daily_credits"
            
            if "table" in (view_mode or ""):
                st.dataframe(filtered, height=CHART_HEIGHT, hide_index=True)
            else:
                if "Bar" in (chart_type or ""):
                    st.altair_chart(
                        create_bar_chart(filtered, "ds", y_col, "region", CHART_HEIGHT, show_percent),
                    )
                else:
                    st.altair_chart(
                        create_line_chart(filtered, "ds", y_col, "region", CHART_HEIGHT),
                    )


    # =============================================================================
    # Page Layout
    # =============================================================================

    render_page_header("# :material/bolt: Compute Dashboard")

    # Row 1: Two metrics
    col1, col2 = st.columns(2)

    with col1:
        account_type_metric()

    with col2:
        instance_type_metric()

    # Row 2: One metric (full width for region breakdown)
    region_metric()


def dashboard_feature_usage():
    """
    API Usage Dashboard Template

    A feature analytics dashboard demonstrating:
    - Segmented control for category selection
    - Multiselect for endpoint filtering
    - Starter kits / presets for quick selection
    - Time series visualization with normalization
    - Metric cards with 28-day deltas
    - Rolling average options

    This template uses synthetic data. Replace generate_api_data()
    with your actual data source (e.g., Snowflake queries, APIs, etc.)
    """



    # =============================================================================
    # Synthetic Data Generation (Replace with your data source)
    # =============================================================================

    # API categories and their endpoints
    API_CATEGORIES = {
        "Users": ["/users", "/users/{id}", "/users/me", "/users/search", "/users/bulk", "/users/export"],
        "Orders": ["/orders", "/orders/{id}", "/orders/create", "/orders/cancel", "/orders/refund", "/orders/status"],
        "Products": ["/products", "/products/{id}", "/products/search", "/products/categories", "/products/inventory"],
        "Analytics": ["/analytics/events", "/analytics/metrics", "/analytics/reports", "/analytics/dashboards"],
    }

    # Starter kits - predefined endpoint selections
    STARTER_KITS = {
        "None": [],
        "Core CRUD": ["/users", "/users/{id}", "/orders", "/orders/{id}"],
        "Search": ["/users/search", "/products/search", "/products/categories"],
        "Analytics": ["/analytics/events", "/analytics/metrics", "/analytics/reports"],
        "High Volume": ["/users", "/products", "/orders", "/analytics/events"],
    }

    ROLLING_OPTIONS = {"Raw": 1, "7-day average": 7, "28-day average": 28}


    def generate_api_data(
        endpoints: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Generate synthetic API usage data.
        
        Replace this function with your actual data source.
        """
        np.random.seed(42)
        
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        records = []
        
        for endpoint in endpoints:
            # Each endpoint has different base traffic and growth
            base = np.random.randint(1000, 50000)
            growth = np.random.uniform(0.0005, 0.003)
            
            for i, dt in enumerate(dates):
                # Base trend with growth
                trend = base * (1 + growth) ** i
                
                # Weekly seasonality (lower on weekends)
                if dt.dayofweek >= 5:
                    trend *= 0.4
                
                # Random noise
                value = trend * np.random.uniform(0.85, 1.15)
                
                records.append({
                    "date": dt,
                    "endpoint": endpoint,
                    "request_count": int(value),
                })
        
        df = pd.DataFrame(records)
        return df


    @st.cache_data(ttl=3600)
    def load_api_data() -> pd.DataFrame:
        """Load all API usage data."""
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=365)
        
        all_endpoints = []
        for endpoints in API_CATEGORIES.values():
            all_endpoints.extend(endpoints)
        
        return generate_api_data(all_endpoints, start_date, end_date)


    def apply_rolling_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Apply rolling average to request data."""
        if window == 1:
            return df
        
        result = df.copy()
        result["request_count"] = (
            result.groupby("endpoint")["request_count"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        return result


    def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize request counts to percentages (share of total per day)."""
        result = df.copy()
        daily_totals = result.groupby("date")["request_count"].transform("sum")
        result["request_count"] = result["request_count"] / daily_totals
        return result


    def calculate_delta(df: pd.DataFrame, endpoint: str) -> tuple[float, float | None]:
        """Calculate 28-day delta for an endpoint."""
        endpoint_data = df[df["endpoint"] == endpoint].sort_values("date")
        
        if len(endpoint_data) < 2:
            return endpoint_data["request_count"].iloc[-1], None
        
        latest = endpoint_data["request_count"].iloc[-1]
        
        if len(endpoint_data) > 28:
            previous = endpoint_data["request_count"].iloc[-29]
        else:
            previous = endpoint_data["request_count"].iloc[0]
        
        delta = latest - previous
        return latest, delta


    # =============================================================================
    # Page Layout
    # =============================================================================

    # Load data
    raw_data = load_api_data()

    # Header
    st.markdown("# API Usage :material/api:")
    st.caption("Select an API category to explore endpoint usage over time.")

    # Category selection (not centered)
    category = st.segmented_control(
        "Select category",
        options=[
            ":material/person: Users",
            ":material/shopping_cart: Orders",
            ":material/inventory_2: Products",
            ":material/analytics: Analytics",
        ],
        default=":material/person: Users",
        label_visibility="collapsed",
    )

    if not category:
        st.warning("Please select a category above.", icon=":material/warning:")
        st.stop()

    # Map display name to category key
    category_map = {
        ":material/person: Users": "Users",
        ":material/shopping_cart: Orders": "Orders",
        ":material/inventory_2: Products": "Products",
        ":material/analytics: Analytics": "Analytics",
    }
    selected_category = category_map[category]

    st.subheader(f"{category} endpoints", divider="gray")

    # Layout: filters on left, chart on right
    filter_col, chart_col = st.columns([1, 2])

    with filter_col:
        # Metric selection
        with st.expander("Metric", expanded=True, icon=":material/analytics:"):
            measure = st.selectbox(
                "Choose a measure",
                ["Request count", "Unique callers", "Error rate"],
                index=0,
                label_visibility="collapsed",
                disabled=True,  # Only one option in this template
                help="In production, connect to different metrics tables",
            )
            
            rolling_label = st.segmented_control(
                "Time aggregation",
                list(ROLLING_OPTIONS.keys()),
                default="7-day average",
                label_visibility="collapsed",
            )
            
            if rolling_label is None:
                st.caption("Please select a time aggregation.")
                st.stop()
            
            rolling_window = ROLLING_OPTIONS[rolling_label]
            
            normalize = st.toggle(
                "Normalize",
                value=False,
                help="Normalize to show percentage share of total requests",
            )
        
        # Starter kits
        with st.expander("Starter kits", expanded=True, icon=":material/auto_awesome:"):
            starter_kit = st.pills(
                "Quick select",
                options=list(STARTER_KITS.keys()),
                default="None",
                label_visibility="collapsed",
            )
        
        # Endpoint selection
        available_endpoints = API_CATEGORIES[selected_category]
        
        # Determine default selection based on starter kit
        if starter_kit and starter_kit != "None":
            default_endpoints = [e for e in STARTER_KITS[starter_kit] if e in available_endpoints]
        else:
            default_endpoints = available_endpoints[:4]  # First 4 endpoints
        
        with st.expander("Endpoints", expanded=True, icon=":material/checklist:"):
            selected_endpoints = st.multiselect(
                "Select endpoints",
                options=available_endpoints,
                default=default_endpoints,
                label_visibility="collapsed",
            )

    # Filter and process data
    if not selected_endpoints:
        with chart_col:
            st.info("Select at least one endpoint to view usage data.", icon=":material/info:")
        st.stop()

    filtered_data = raw_data[raw_data["endpoint"].isin(selected_endpoints)].copy()
    filtered_data = apply_rolling_average(filtered_data, rolling_window)

    if normalize:
        filtered_data = normalize_data(filtered_data)

    with chart_col:
        # Latest metrics
        with st.expander("Latest numbers", expanded=True, icon=":material/numbers:"):
            metrics_row = st.container(horizontal=True)
            
            for endpoint in selected_endpoints:
                latest, delta = calculate_delta(filtered_data, endpoint)
                
                if normalize:
                    value_str = f"{latest:.2%}"
                    delta_str = f"{delta:+.2%}" if delta is not None else None
                else:
                    value_str = f"{latest:,.0f}"
                    delta_str = f"{delta:+,.0f}" if delta is not None else None
                
                metrics_row.metric(
                    label=endpoint,
                    value=value_str,
                    delta=delta_str,
                    border=True,
                )
        
        # Time series chart
        with st.expander("Time series", expanded=True, icon=":material/show_chart:"):
            y_format = ".1%" if normalize else ",.0f"
            y_title = "Share of requests" if normalize else "Request count"
            
            chart = (
                alt.Chart(filtered_data)
                .mark_line()
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("request_count:Q", title=y_title, axis=alt.Axis(format=y_format)),
                    color=alt.Color("endpoint:N", title="Endpoint", legend=alt.Legend(orient="bottom")),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                        alt.Tooltip("endpoint:N", title="Endpoint"),
                        alt.Tooltip("request_count:Q", title="Requests", format=y_format),
                    ],
                )
                .properties(height=450)
                .interactive()
            )
            
            st.altair_chart(chart)

    # Raw data section
    with st.expander("Raw data", expanded=False, icon=":material/table:"):
        display_df = filtered_data.copy()
        if normalize:
            display_df["request_count"] = display_df["request_count"].apply(lambda x: f"{x:.2%}")
        st.dataframe(display_df, hide_index=True)

def dashboard_weather_seattle():
   
    url = "https://raw.githubusercontent.com/vega/vega/refs/heads/main/docs/data/seattle-weather.csv"
   
    full_df= pd.read_csv(url)
    #full_df = vega_datasets.data("seattle_weather")




    """
    # Seattle Weather

    Let's explore the [classic Seattle Weather
    dataset](https://altair-viz.github.io/case_studies/exploring-weather.html)!
    """

    ""  # Add a little vertical space. Same as st.write("").
    ""

    """
    ## 2015 Summary
    """

  
    full_df["date"]= pd.to_datetime(full_df["date"])
    df_2015 = full_df[full_df["date"].dt.year == 2015]
    df_2014 = full_df[full_df["date"].dt.year == 2014]

    max_temp_2015 = df_2015["temp_max"].max()
    max_temp_2014 = df_2014["temp_max"].max()

    min_temp_2015 = df_2015["temp_min"].min()
    min_temp_2014 = df_2014["temp_min"].min()

    max_wind_2015 = df_2015["wind"].max()
    max_wind_2014 = df_2014["wind"].max()

    min_wind_2015 = df_2015["wind"].min()
    min_wind_2014 = df_2014["wind"].min()

    max_prec_2015 = df_2015["precipitation"].max()
    max_prec_2014 = df_2014["precipitation"].max()

    min_prec_2015 = df_2015["precipitation"].min()
    min_prec_2014 = df_2014["precipitation"].min()


    with st.container(horizontal=True, gap="medium"):
        cols = st.columns(2, gap="medium", width=300)

        with cols[0]:
            st.metric(
                "Max temperature",
                f"{max_temp_2015:0.1f}C",
                delta=f"{max_temp_2015 - max_temp_2014:0.1f}C",
                width="content",
            )

        with cols[1]:
            st.metric(
                "Min temperature",
                f"{min_temp_2015:0.1f}C",
                delta=f"{min_temp_2015 - min_temp_2014:0.1f}C",
                width="content",
            )

        cols = st.columns(2, gap="medium", width=300)

        with cols[0]:
            st.metric(
                "Max precipitation",
                f"{max_prec_2015:0.1f}mm",
                delta=f"{max_prec_2015 - max_prec_2014:0.1f}mm",
                width="content",
            )

        with cols[1]:
            st.metric(
                "Min precipitation",
                f"{min_prec_2015:0.1f}mm",
                delta=f"{min_prec_2015 - min_prec_2014:0.1f}mm",
                width="content",
            )

        cols = st.columns(2, gap="medium", width=300)

        with cols[0]:
            st.metric(
                "Max wind",
                f"{max_wind_2015:0.1f}m/s",
                delta=f"{max_wind_2015 - max_wind_2014:0.1f}m/s",
                width="content",
            )

        with cols[1]:
            st.metric(
                "Min wind",
                f"{min_wind_2015:0.1f}m/s",
                delta=f"{min_wind_2015 - min_wind_2014:0.1f}m/s",
                width="content",
            )

        weather_icons = {
            "sun": "sunny",
            "snow": "weather_snowy",
            "rain": "rainy",
            "fog": "foggy",
            "drizzle": "rainy",
        }

        cols = st.columns(2, gap="large")

        with cols[0]:
            weather_name = (
                full_df["weather"].value_counts().head(1).reset_index()["weather"][0]
            )
            st.metric(
                "Most common weather",
                f":material/{weather_icons[weather_name]}: {weather_name.upper()}",
            )

        with cols[1]:
            weather_name = (
                full_df["weather"].value_counts().tail(1).reset_index()["weather"][0]
            )
            st.metric(
                "Least common weather",
                f":material/{weather_icons[weather_name]}: {weather_name.upper()}",
            )

    ""
    ""

    """
    ## Compare different years
    """

    YEARS = full_df["date"].dt.year.unique()
    selected_years = st.pills(
        "Years to compare", YEARS, default=YEARS, selection_mode="multi"
    )

    if not selected_years:
        st.warning("You must select at least 1 year.", icon=":material/warning:")

    df = full_df[full_df["date"].dt.year.isin(selected_years)]

    cols = st.columns([3, 1])

    with cols[0].container(border=True, height="stretch"):
        "### 🌡️ Temperature"

        st.altair_chart(
            alt.Chart(df)
            .mark_bar(width=1)
            .encode(
                alt.X("monthdate(date):T").title("date"),
                alt.Y("temp_max:Q").title("temperature range (C)"),
                alt.Y2("temp_min:Q"),
                alt.Color("year(date):N").title("year"),
                alt.XOffset("year(date):N"),
                tooltip=[
                    alt.Tooltip("monthdate(date):T", title="Date"),
                    alt.Tooltip("temp_max:Q", title="Max Temp (C)"),
                    alt.Tooltip("temp_min:Q", title="Min Temp (C)"),
                    alt.Tooltip("year(date):N", title="Year"),
                ],
            )
            .configure_legend(orient="bottom")
        )

    with cols[1].container(border=True, height="stretch"):
        "### Weather distribution"

        st.altair_chart(
            alt.Chart(df)
            .mark_arc()
            .encode(
                alt.Theta("count()"),
                alt.Color("weather:N"),
            )
            .configure_legend(orient="bottom")
        )


    cols = st.columns(2)

    with cols[0].container(border=True, height="stretch"):
        "### 💨 Wind"

        # Prepare data for st.line_chart - pivot by year
        wind_df = df.copy()
        wind_df["month_day"] = wind_df["date"].dt.strftime("%m-%d")
        wind_df["year"] = wind_df["date"].dt.year
        
        # Calculate 14-day rolling average per year
        wind_pivot = wind_df.pivot_table(
            index="month_day", 
            columns="year", 
            values="wind", 
            aggfunc="mean"
        ).sort_index()
        
        st.line_chart(wind_pivot, height=300)

    with cols[1].container(border=True, height="stretch"):
        "### 🌧️ Precipitation"

        st.altair_chart(
            alt.Chart(df)
            .mark_bar()
            .encode(
                alt.X("month(date):O").title("month"),
                alt.Y("sum(precipitation):Q").title("precipitation (mm)"),
                alt.Color("year(date):N").title("year"),
                tooltip=[
                    alt.Tooltip("month(date):O", title="Month"),
                    alt.Tooltip("sum(precipitation):Q", title="Precipitation (mm)"),
                    alt.Tooltip("year(date):N", title="Year"),
                ],
            )
            .configure_legend(orient="bottom")
        )

    cols = st.columns(2)

    with cols[0].container(border=True, height="stretch"):
        "### Monthly weather breakdown"
        ""

        st.altair_chart(
            alt.Chart(df)
            .mark_bar()
            .encode(
                alt.X("month(date):O", title="month"),
                alt.Y("count():Q", title="days").stack("normalize"),
                alt.Color("weather:N"),
            )
            .configure_legend(orient="bottom")
        )

    with cols[1].container(border=True, height="stretch"):
        "### Raw data"

        st.dataframe(df)
def dashboard_stock_peers():

  

    """
    # :material/query_stats: Stock peer analysis

    Easily compare stocks against others in their peer group.
    """

    ""  # Add some space.

    cols = st.columns([1, 3])
    # Will declare right cell later to avoid showing it when no data.

    STOCKS = [
        "AAPL",
        "ABBV",
        "ACN",
        "ADBE",
        "ADP",
        "AMD",
        "AMGN",
        "AMT",
        "AMZN",
        "APD",
        "AVGO",
        "AXP",
        "BA",
        "BK",
        "BKNG",
        "BMY",
        "BRK.B",
        "BSX",
        "C",
        "CAT",
        "CI",
        "CL",
        "CMCSA",
        "COST",
        "CRM",
        "CSCO",
        "CVX",
        "DE",
        "DHR",
        "DIS",
        "DUK",
        "ELV",
        "EOG",
        "EQR",
        "FDX",
        "GD",
        "GE",
        "GILD",
        "GOOG",
        "GOOGL",
        "HD",
        "HON",
        "HUM",
        "IBM",
        "ICE",
        "INTC",
        "ISRG",
        "JNJ",
        "JPM",
        "KO",
        "LIN",
        "LLY",
        "LMT",
        "LOW",
        "MA",
        "MCD",
        "MDLZ",
        "META",
        "MMC",
        "MO",
        "MRK",
        "MSFT",
        "NEE",
        "NFLX",
        "NKE",
        "NOW",
        "NVDA",
        "ORCL",
        "PEP",
        "PFE",
        "PG",
        "PLD",
        "PM",
        "PSA",
        "REGN",
        "RTX",
        "SBUX",
        "SCHW",
        "SLB",
        "SO",
        "SPGI",
        "T",
        "TJX",
        "TMO",
        "TSLA",
        "TXN",
        "UNH",
        "UNP",
        "UPS",
        "V",
        "VZ",
        "WFC",
        "WM",
        "WMT",
        "XOM",
    ]

    DEFAULT_STOCKS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "TSLA", "META"]


    def stocks_to_str(stocks):
        return ",".join(stocks)


    if "tickers_input" not in st.session_state:
        st.session_state.tickers_input = st.query_params.get(
            "stocks", stocks_to_str(DEFAULT_STOCKS)
        ).split(",")


    # Callback to update query param when input changes
    def update_query_param():
        if st.session_state.tickers_input:
            st.query_params["stocks"] = stocks_to_str(st.session_state.tickers_input)
        else:
            st.query_params.pop("stocks", None)


    top_left_cell = cols[0].container(
        border=True, height="stretch", vertical_alignment="center"
    )

    with top_left_cell:
        # Selectbox for stock tickers
        tickers = st.multiselect(
            "Stock tickers",
            options=sorted(set(STOCKS) | set(st.session_state.tickers_input)),
            default=st.session_state.tickers_input,
            placeholder="Choose stocks to compare. Example: NVDA",
            accept_new_options=True,
        )

    # Time horizon selector
    horizon_map = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "5 Years": "5y",
        "10 Years": "10y",
        "20 Years": "20y",
    }

    with top_left_cell:
        # Buttons for picking time horizon
        horizon = st.pills(
            "Time horizon",
            options=list(horizon_map.keys()),
            default="6 Months",
        )

    tickers = [t.upper() for t in tickers]

    # Update query param when text input changes
    if tickers:
        st.query_params["stocks"] = stocks_to_str(tickers)
    else:
        # Clear the param if input is empty
        st.query_params.pop("stocks", None)

    if not tickers:
        top_left_cell.info("Pick some stocks to compare", icon=":material/info:")
        st.stop()


    right_cell = cols[1].container(
        border=True, height="stretch", vertical_alignment="center"
    )


    @st.cache_resource(show_spinner=False, ttl="6h")
    def load_data(tickers, period):
        tickers_obj = yf.Tickers(tickers)
        data = tickers_obj.history(period=period)
        if data is None:
            raise RuntimeError("YFinance returned no data.")
        return data["Close"]


    # Load the data
    try:
        data = load_data(tickers, horizon_map[horizon])
    except yf.exceptions.YFRateLimitError as e:
        st.warning("YFinance is rate-limiting us :(\nTry again later.")
        load_data.clear()  # Remove the bad cache entry.
        st.stop()
    data=data[1:]
    empty_columns = data.columns[data.isna().all()].tolist()
   
    if empty_columns:
        st.error(f"Error loading data for the tickers: {', '.join(empty_columns)}.")
        st.stop()

    # Normalize prices (start at 1)
    normalized = data.div(data.iloc[0])

    latest_norm_values = {normalized[ticker].iat[-1]: ticker for ticker in tickers}
    max_norm_value = max(latest_norm_values.items())
    min_norm_value = min(latest_norm_values.items())

    bottom_left_cell = cols[0].container(
        border=True, height="stretch", vertical_alignment="center"
    )
    delta = "n/a" if not np.isfinite(max_norm_value[0]) else f"{round(max_norm_value[0] * 100)}%"
    with bottom_left_cell:
        cols = st.columns(2)
        cols[0].metric(
            "Best stock",
            max_norm_value[1],
            delta=delta ,
            width="content",
        )
        cols[1].metric(
            "Worst stock",
            min_norm_value[1],
            delta=delta ,
            width="content",
        )


    # Plot normalized prices
    with right_cell:
        st.altair_chart(
            alt.Chart(
                normalized.reset_index().melt(
                    id_vars=["Date"], var_name="Stock", value_name="Normalized price"
                )
            )
            .mark_line()
            .encode(
                alt.X("Date:T"),
                alt.Y("Normalized price:Q").scale(zero=False),
                alt.Color("Stock:N"),
            )
            .properties(height=400)
        )

    ""
    ""

    # Plot individual stock vs peer average
    """
    ## Individual stocks vs peer average

    For the analysis below, the "peer average" when analyzing stock X always
    excludes X itself.
    """

    if len(tickers) <= 1:
        st.warning("Pick 2 or more tickers to compare them")
        st.stop()

    NUM_COLS = 4
    cols = st.columns(NUM_COLS)

    for i, ticker in enumerate(tickers):
        # Calculate peer average (excluding current stock)
        peers = normalized.drop(columns=[ticker])
        peer_avg = peers.mean(axis=1)

        # Create DataFrame with peer average.
        plot_data = pd.DataFrame(
            {
                "Date": normalized.index,
                ticker: normalized[ticker],
                "Peer average": peer_avg,
            }
        ).melt(id_vars=["Date"], var_name="Series", value_name="Price")

        chart = (
            alt.Chart(plot_data)
            .mark_line()
            .encode(
                alt.X("Date:T"),
                alt.Y("Price:Q").scale(zero=False),
                alt.Color(
                    "Series:N",
                    scale=alt.Scale(domain=[ticker, "Peer average"], range=["red", "gray"]),
                    legend=alt.Legend(orient="bottom"),
                ),
                alt.Tooltip(["Date", "Series", "Price"]),
            )
            .properties(title=f"{ticker} vs peer average", height=300)
        )

        cell = cols[(i * 2) % NUM_COLS].container(border=True)
        cell.write("")
        cell.altair_chart(chart)

        # Create Delta chart
        plot_data = pd.DataFrame(
            {
                "Date": normalized.index,
                "Delta": normalized[ticker] - peer_avg,
            }
        )

        chart = (
            alt.Chart(plot_data)
            .mark_area()
            .encode(
                alt.X("Date:T"),
                alt.Y("Delta:Q").scale(zero=False),
            )
            .properties(title=f"{ticker} minus peer average", height=300)
        )

        cell = cols[(i * 2 + 1) % NUM_COLS].container(border=True)
        cell.write("")
        cell.altair_chart(chart)

    ""
    ""

    """
    ## Raw data
    """

    data

def main():
    st.header("Streamlit Dashboard App Templates")
    st.info("These script contains all ready-to-use dashboard templates for Streamlit. Each template demonstrates best practices for building data-driven dashboards with modern UI patterns.")
    with st.expander("COMPANIES"):
        dashboard_companies()
   
    
   
    with st.expander("METRICS"): 
        dashboard_metrics()
   
    with st.expander("COMPUTE"):
        dashboard_compute()
    
   
    with st.expander("FEATURE USAGE"):
        dashboard_feature_usage()
   
    with st.expander("WEATHER"):
        dashboard_weather_seattle()
    
  
    with st.expander("STOCK PEERS"):
        
        dashboard_stock_peers()

    st.info("Source: https://github.com/streamlit/agent-skills/tree/main/developing-with-streamlit/templates/apps")
if __name__ == "__main__":
    main()
  
