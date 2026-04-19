# streamlit run loessviewer.py

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from skmisc.loess import loess
    SKMISC_AVAILABLE = True
except Exception:
    SKMISC_AVAILABLE = False

st.set_page_config(page_title="LOESS Viewer", layout="wide")
st.title("LOESS Viewer")


def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        for sep in [",", ";", "\t"]:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        xls = pd.ExcelFile(uploaded_file)
        if len(xls.sheet_names) == 1:
            return pd.read_excel(uploaded_file, sheet_name=xls.sheet_names[0])

        sheet = st.selectbox("Sheet", xls.sheet_names)
        return pd.read_excel(uploaded_file, sheet_name=sheet)

    raise ValueError("Unsupported file type")


def to_numeric_time(series):
    s = series.copy()

    if np.issubdtype(s.dtype, np.datetime64):
        t = (s - s.min()).dt.total_seconds() / 86400.0
        return t.to_numpy(dtype=float), pd.to_datetime(s), True

    s_num = pd.to_numeric(s, errors="coerce")
    return s_num.to_numpy(dtype=float), s, False

def loess_smooth(x,y,span):
    
    if 1==1:
        l = loess(x,y)
        
        
        # MODEL and CONTROL. Essential for replicating the results from the R script.
        #
        # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess_model.html#skmisc.loess.loess_model
        # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess_control.html#skmisc.loess.loess_control
    
        l.model.span = span
        l.model.degree = 1
        l.control.iterations = 1 #it # must be 1 for replicating the R-script
        l.control.surface = "direct"
        if len(y)<100:
            l.control.statistics = "exact"
        else:
            l.control.statistics = "approximate"   # or "none" if supported in your version


        l.fit()
        pred = l.predict(x, stderror=False)
        lowess = pred.values
    else:
        lowess=y
    return lowess

def loess_smooth_(x, y, span):
    if SKMISC_AVAILABLE:
        model = loess(x, y, span=span, degree=1)
        model.fit()
        return model.predict(x).values

    # fallback
    n = len(x)
    r = max(2, int(np.ceil(span * n)))
    yest = np.zeros(n)

    for i in range(n):
        distances = np.abs(x - x[i])
        idx = np.argsort(distances)[:r]
        dmax = distances[idx[-1]]

        if dmax == 0:
            yest[i] = y[i]
            continue

        w = (1 - (distances[idx] / dmax) ** 3) ** 3
        X = np.vstack([np.ones(len(idx)), x[idx]]).T
        W = np.diag(w)

        beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y[idx])
        yest[i] = beta[0] + beta[1] * x[i]

    return yest

def main():
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if uploaded_file is None:
        st.stop()

    try:
        df = read_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    if df.empty:
        st.error("File is empty")
        st.stop()
    with st.expander("Data preview"):
        st.dataframe(df.head(20), width='stretch')

    all_columns = df.columns.tolist()

    col1, col2, col3 = st.columns(3)

    with col1:
        filter_col = st.selectbox("Filter column", ["None"] + all_columns)

    with col2:
        if filter_col != "None":
            unique_vals = df[filter_col].dropna().astype(str).unique().tolist()
            unique_vals = sorted(unique_vals)
            filter_val = st.selectbox("Filter value", ["All"] + unique_vals)
        else:
            filter_val = "All"
            st.selectbox("Filter value", ["All"], disabled=True)

    with col3:
        x_col = st.selectbox("X axis", all_columns)

    df_plot = df.copy()

    if filter_col != "None" and filter_val != "All":
        df_plot = df_plot[df_plot[filter_col].astype(str) == str(filter_val)].copy()

    if df_plot.empty:
        st.warning("No rows left after filtering")
        st.stop()

    numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != x_col]
    col1, col2, col3,col4 = st.columns(4)
    with col1:
        y_col = st.selectbox(
            "Columns for LOESS",
            numeric_cols,
            
            help="Choose columns to apply LOESS smoothing to. The raw data will also be plotted for comparison.",
        )

        if not y_col:
            st.warning("Choose at least 1 column")
            st.stop()
    with col2:
        window_1 = st.number_input("Window 1", min_value=1, value=7, step=1)
    with col3:
        window_2 = st.number_input("Window 2 (0 for None)", min_value=0, value=365, step=1)
    with col4:
        window_3 = st.number_input("Window 3 (0 for None)", min_value=0, value=0, step=1)

    windows=[window_1,window_2,window_3]
    df_plot = df_plot.sort_values(by=x_col).copy()

    x_numeric, x_display, is_datetime = to_numeric_time(df_plot[x_col])

    valid_x = ~np.isnan(x_numeric)
    df_plot = df_plot.loc[valid_x].copy()
    x_numeric = x_numeric[valid_x]
    x_display = x_display.loc[valid_x] if hasattr(x_display, "loc") else x_display[valid_x]
    first = True
    fig = go.Figure()

    


    for window in windows:
        if window !=0:
            print (f"Processing window {window}") 
        
            y = pd.to_numeric(df_plot[y_col], errors="coerce").to_numpy(dtype=float)
            valid = ~np.isnan(y) & ~np.isnan(x_numeric)

            if valid.sum() < 3:
                st.warning(f"Too few valid rows for {y_col}")
                continue

            x_use = x_numeric[valid]
            y_use = y[valid]
            x_show = x_display.loc[valid] if hasattr(x_display, "loc") else x_display[valid]

            span = 1.412 * window / len(x_use)
            span = min(max(span, 0.001), 1.0)

            y_loess = loess_smooth(x_use, y_use, span)
            print (y_loess)
            if first:
                fig.add_trace(
                    go.Scatter(
                        x=x_show,
                        y=y_use,
                        mode="markers",
                        name=f"{y_col} raw",
                        opacity=0.35,
                    )
                )
                first=False
            fig.add_trace(
                go.Scatter(
                    x=x_show,
                    y=y_loess,
                    mode="lines",
                    name=f"{y_col} loess {window}",
                    line=dict(width=3),
                )
            )

    st.caption(
        "Span formula: 1.412 × window / len(t)"
        + ("  |  Using skmisc.loess" if SKMISC_AVAILABLE else "  |  Using fallback LOESS")
    )

    fig.update_layout(
        height=700,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title=x_col,
        yaxis_title="Value",
        legend_title="Series",
    )

    st.plotly_chart(fig, width='stretch')

if __name__=="__main__":
    print("Start")
    main()