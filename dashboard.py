"""
dashboard.py — Results viewer for geval
Run: streamlit run dashboard.py
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="geval Results", layout="wide")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(path="results.xlsx"):
    results = pd.read_excel(path, sheet_name="Results")
    claims  = pd.read_excel(path, sheet_name="Claim Breakdown")
    return results, claims

try:
    df, claims_df = load_data()
except FileNotFoundError:
    st.error("results.xlsx not found. Run `python run_tests.py` first.")
    st.stop()

# ---------------------------------------------------------------------------
# Header + summary metrics
# ---------------------------------------------------------------------------

st.title("geval — Completeness Evaluation Results")

total  = len(df)
passed = (df["pass_fail"] == "PASS").sum()
failed = (df["pass_fail"] == "FAIL").sum()
avg_score = df["score"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Tests", total)
c2.metric("Passed", passed, delta=f"{passed/total:.0%}")
c3.metric("Failed", failed, delta=f"-{failed/total:.0%}", delta_color="inverse")
c4.metric("Avg Score", f"{avg_score:.2f}")

st.divider()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)
with col1:
    doc_filter = st.multiselect("Document type", sorted(df["document_type"].unique()))
with col2:
    scenario_filter = st.multiselect("Scenario", sorted(df["scenario_type"].unique()))
with col3:
    status_filter = st.multiselect("Pass / Fail", ["PASS", "FAIL"])

filtered = df.copy()
if doc_filter:
    filtered = filtered[filtered["document_type"].isin(doc_filter)]
if scenario_filter:
    filtered = filtered[filtered["scenario_type"].isin(scenario_filter)]
if status_filter:
    filtered = filtered[filtered["pass_fail"].isin(status_filter)]

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def colour_row(row):
    colour = "#d4edda" if row["pass_fail"] == "PASS" else "#f8d7da"
    return [f"background-color: {colour}"] * len(row)

display_cols = ["test_id", "document_type", "scenario_type",
                "expected_label", "actual_label", "score", "pass_fail"]

st.dataframe(
    filtered[display_cols].style.apply(colour_row, axis=1),
    use_container_width=True,
    height=280,
)

st.divider()

# ---------------------------------------------------------------------------
# Detail panel
# ---------------------------------------------------------------------------

st.subheader("Inspect a test case")

selected_id = st.selectbox(
    "Select test ID",
    options=filtered["test_id"].tolist(),
    format_func=lambda x: f"{x}  —  {filtered.loc[filtered['test_id']==x, 'document_type'].values[0]}  |  {filtered.loc[filtered['test_id']==x, 'scenario_type'].values[0]}"
)

if selected_id:
    row = df[df["test_id"] == selected_id].iloc[0]

    # Status badge
    badge = "✅ PASS" if row["pass_fail"] == "PASS" else "❌ FAIL"
    st.markdown(f"### {selected_id} &nbsp; {badge}")

    # Key info
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Score",          f"{row['score']:.2f}")
    m2.metric("Expected",       row["expected_label"])
    m3.metric("Predicted",      row["actual_label"])
    m4.metric("Critical claims",row["critical_claims"])
    m5.metric("Total claims",   row["total_claims"])

    # Model response + document side by side
    left, right = st.columns(2)
    with left:
        st.markdown("**Model Response**")
        st.text_area("", value=row["model_response"], height=220, label_visibility="collapsed")
    with right:
        st.markdown("**Missing Critical Claims**")
        missing = str(row.get("missing_critical", "") or "")
        if missing.strip():
            for claim in missing.split(";"):
                st.error(claim.strip())
        else:
            st.success("None — all critical claims covered.")

        st.markdown("**Missing Supporting Claims**")
        missing_sup = str(row.get("missing_supporting", "") or "")
        if missing_sup.strip():
            for claim in missing_sup.split(";"):
                st.warning(claim.strip())
        else:
            st.success("None.")

    # Claim breakdown table
    st.markdown("**Claim-level breakdown**")
    case_claims = claims_df[claims_df["test_id"] == selected_id][
        ["claim_id", "coverage_score", "coverage_label", "reason"]
    ]

    def colour_claim(row):
        colours = {2: "#d4edda", 1: "#fff3cd", 0: "#f8d7da"}
        c = colours.get(row["coverage_score"], "white")
        return [f"background-color: {c}"] * len(row)

    if not case_claims.empty:
        st.dataframe(
            case_claims.style.apply(colour_claim, axis=1),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No claim breakdown available for this test case.")
