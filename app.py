import io
from typing import Optional

import pandas as pd
import streamlit as st
from pandas.api import types as ptypes

st.set_page_config(page_title="Data Cleaning App", layout="wide")

st.title("🧹 Data Cleaning App")
st.write("Upload a CSV or Excel file to inspect and clean missing values and duplicates.")

# Helpers
@st.cache_data
def read_file(uploaded) -> pd.DataFrame:
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    else:
        return pd.read_excel(uploaded)

def reset_session_df(df: pd.DataFrame):
    # store original and working copy
    st.session_state["original_df"] = df.copy()
    st.session_state["df"] = df.copy()

def download_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cleaned")
    return buffer.getvalue()

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    dup = df.duplicated().sum()
    types = df.dtypes.astype(str)
    summary = pd.DataFrame({
        "dtype": types,
        "missing": miss,
        "missing_pct": miss_pct
    })
    summary.loc["__duplicates"] = ["", dup, ""]  # convenience row
    return summary

# Upload area in the sidebar
with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    # Global quick options
    st.markdown("### Quick actions")
    default_remove_duplicates = st.checkbox("Remove duplicates on upload", value=False)
    default_dropna_on_upload = st.checkbox("Drop rows with any missing values on upload", value=False)

# Load file into session_state
if uploaded_file:
    try:
        df_in = read_file(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    # initialize session state on first load or when a different file is uploaded
    if ("uploaded_name" not in st.session_state) or (st.session_state.get("uploaded_name") != uploaded_file.name):
        st.session_state["uploaded_name"] = uploaded_file.name
        reset_session_df(df_in)
        if default_remove_duplicates:
            st.session_state["df"] = st.session_state["df"].drop_duplicates().reset_index(drop=True)
        if default_dropna_on_upload:
            st.session_state["df"] = st.session_state["df"].dropna().reset_index(drop=True)

# If no upload, show instructions
if "df" not in st.session_state:
    st.info("Upload a CSV or Excel file (top-left) to start cleaning.")
    st.stop()

df = st.session_state["df"]
original_df = st.session_state["original_df"]

# Layout: left = data & stats, right = cleaning controls
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Original Data (first 10 rows)")
    st.dataframe(original_df.head(10))

    st.subheader("Working Data (first 10 rows)")
    st.dataframe(df.head(10))

    st.subheader("Summary")
    stats = summary_stats(df)
    st.dataframe(stats)

with right_col:
    st.subheader("Missing value handling")

    # select columns to operate on
    cols = list(df.columns)
    chosen_cols = st.multiselect("Select columns to handle (defaults to all with missing)", options=cols,
                                 default=[c for c in cols if df[c].isnull().any()])

    # Build per-column strategy UI
    strategies = {}
    for col in chosen_cols:
        st.markdown(f"**{col}** — dtype: {df[col].dtype}")
        if ptypes.is_numeric_dtype(df[col]):
            strat = st.selectbox(f"Numeric strategy for {col}", options=["mean", "median", "constant"], key=f"{col}_num")
            if strat == "constant":
                const = st.text_input(f"Constant value for {col} (leave blank for 0)", value="", key=f"{col}_const")
                strategies[col] = ("numeric_constant", const)
            else:
                strategies[col] = (strat, None)
        else:
            strat = st.selectbox(f"Categorical strategy for {col}", options=["mode", "ffill", "bfill", "constant"], key=f"{col}_cat")
            if strat == "constant":
                const = st.text_input(f"Constant value for {col} (leave blank for 'Unknown')", value="", key=f"{col}_const_cat")
                strategies[col] = ("cat_constant", const)
            else:
                strategies[col] = (strat, None)

    apply_impute = st.button("Apply missing-value strategies")

    st.markdown("---")
    st.subheader("Duplicates")
    if st.button("Show duplicate sample"):
        dup_idx = df[df.duplicated(keep=False)].index
        if len(dup_idx):
            st.dataframe(df.loc[dup_idx].head(20))
        else:
            st.info("No duplicate rows found.")
    if st.button("Remove duplicate rows"):
        before = len(st.session_state["df"])
        st.session_state["df"] = st.session_state["df"].drop_duplicates().reset_index(drop=True)
        after = len(st.session_state["df"])
        st.success(f"Removed {before - after} duplicate rows.")

    st.markdown("---")
    st.subheader("Other actions")
    if st.button("Reset to original upload"):
        reset_session_df(original_df)
        st.success("Reset working data to original uploaded data.")

    if st.button("Drop rows with any missing values"):
        before = len(st.session_state["df"])
        st.session_state["df"] = st.session_state["df"].dropna().reset_index(drop=True)
        after = len(st.session_state["df"])
        st.success(f"Dropped {before - after} rows containing missing values.")

# Apply imputation when requested (outside sidebar so it can update)
if uploaded_file and apply_impute:
    df = st.session_state["df"]
    changed = False
    for col, (strategy, param) in strategies.items():
        if strategy in ("mean", "median") and ptypes.is_numeric_dtype(df[col]):
            if strategy == "mean":
                try:
                    fill_val = df[col].mean()
                except Exception:
                    fill_val = 0
            else:
                try:
                    fill_val = df[col].median()
                except Exception:
                    fill_val = 0
            df[col] = df[col].fillna(fill_val)
            changed = True
        elif strategy == "numeric_constant":
            try:
                # try cast to numeric; fallback to 0
                fill_val = float(param) if param != "" else 0.0
            except Exception:
                fill_val = 0.0
            df[col] = df[col].fillna(fill_val)
            changed = True
        elif strategy == "mode":
            try:
                m = df[col].mode()
                fill_val = m.iloc[0] if not m.empty else ""
            except Exception:
                fill_val = ""
            df[col] = df[col].fillna(fill_val)
            changed = True
        elif strategy in ("ffill", "bfill"):
            df[col] = df[col].fillna(method=strategy)
            changed = True
        elif strategy == "cat_constant":
            fill_val = param if param != "" else "Unknown"
            df[col] = df[col].fillna(fill_val)
            changed = True
        else:
            # unknown strategy or mismatch dtype: skip
            st.warning(f"Skipped {col}: incompatible strategy or dtype.")
    if changed:
        st.session_state["df"] = df.reset_index(drop=True)
        st.success("Applied missing-value strategies.")

# Update local variable
df = st.session_state["df"]

# Show before/after summary comparison
st.markdown("## Before / After")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Before upload")
    st.write(f"Rows: {len(original_df)}")
    st.write(f"Columns: {len(original_df.columns)}")
    st.write("Missing per column:")
    st.dataframe(original_df.isnull().sum().to_frame("missing_before"))
with col2:
    st.markdown("### After cleaning (working)")
    st.write(f"Rows: {len(df)}")
    st.write(f"Columns: {len(df.columns)}")
    st.write("Missing per column:")
    st.dataframe(df.isnull().sum().to_frame("missing_after"))

# Download
st.markdown("---")
st.subheader("Download cleaned data")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv_bytes, file_name="cleaned_data.csv", mime="text/csv")
try:
    excel_bytes = download_excel_bytes(df)
    st.download_button("Download Excel", excel_bytes, file_name="cleaned_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
except Exception:
    st.info("Excel download requires openpyxl; CSV is available.")

st.caption("Tip: Use 'Reset to original upload' to discard changes and start again.")