import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f4f7fb; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1250px; }
.hero {
  background: linear-gradient(120deg, #101c3a 0%, #183c70 55%, #2a6b91 100%);
  padding: 2.1rem 2.2rem; border-radius: 22px; color: white;
  box-shadow: 0 12px 34px rgba(17, 38, 74, .18); margin-bottom: 1.3rem;
}
.hero h1 { color: white; font-family: 'Manrope', sans-serif; font-size: 2.15rem; margin: .2rem 0 .45rem 0; }
.hero p { color: #d8e7ff; font-size: 1rem; margin-bottom: .2rem; }
.eyebrow { color: #9fd7ff; text-transform: uppercase; font-size: .75rem; letter-spacing: .16em; font-weight: 700; }
.metric-card {
  background: white; padding: 1.1rem 1.25rem; border-radius: 16px;
  border: 1px solid #e2e9f2; box-shadow: 0 4px 16px rgba(20, 45, 80, .04);
}
.metric-label { color: #61718b; font-size: .82rem; font-weight: 600; }
.metric-value { color: #172746; font-family: 'Manrope', sans-serif; font-size: 1.75rem; font-weight: 800; margin-top: .25rem; }
.metric-note { color: #77859b; font-size: .76rem; margin-top: .15rem; }
.panel {
  background: white; padding: 1.2rem 1.3rem; border-radius: 16px;
  border: 1px solid #e2e9f2; margin-bottom: 1rem;
}
.small-muted { color: #697992; font-size: .9rem; }
div.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, #1769aa, #2785b8); color: white;
  border: 0; border-radius: 10px; font-weight: 700; padding: .65rem 1.2rem;
}
div.stButton > button[kind="primary"]:hover { background: #10598f; color: white; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "Study_Hours", "Attendance", "Previous_Marks",
    "Assignment_Score", "Internal_Marks", "Sleep_Hours"
]
DISPLAY = {
    "Study_Hours": "Study hours per day",
    "Attendance": "Attendance (%)",
    "Previous_Marks": "Previous marks (%)",
    "Assignment_Score": "Assignment score (%)",
    "Internal_Marks": "Internal marks (%)",
    "Sleep_Hours": "Sleep hours per night",
}

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("student_performance_dataset.csv")
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def train_models(csv_bytes):
    import io
    df = pd.read_csv(io.BytesIO(csv_bytes))
    missing = [c for c in FEATURES + ["Result"] if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    df = df.dropna(subset=FEATURES + ["Result"]).copy()
    X = df[FEATURES]
    y_text = df["Result"].astype(str).str.strip().str.title()
    valid = y_text.isin(["Pass", "Fail"])
    X, y_text = X.loc[valid], y_text.loc[valid]
    y = (y_text == "Pass").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    models = {"Logistic Regression": lr, "Decision Tree": dt}
    metrics = {}
    for name, model in models.items():
        pred = model.predict(X_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "matrix": confusion_matrix(y_test, pred, labels=[0, 1]),
            "report": classification_report(
                y_test, pred, labels=[0, 1],
                target_names=["Fail", "Pass"], output_dict=True, zero_division=0
            )
        }
    return df, models, metrics, len(X_train), len(X_test)

df, load_error = load_dataset()

# Header
st.markdown("""
<div class="hero">
  <div class="eyebrow">Machine Learning • College Project</div>
  <h1>🎓 Student Performance Predictor</h1>
  <p>An interactive demonstration that compares Logistic Regression and Decision Tree models to estimate a student's result.</p>
</div>
""", unsafe_allow_html=True)

if df is None:
    st.error("Dataset not found. Put student_performance_dataset.csv in the same folder as app.py.")
    st.code("student_performance_website/\\n  app.py\\n  student_performance_dataset.csv\\n  requirements.txt")
    st.stop()

try:
    df, models, metrics, train_count, test_count = train_models(df.to_csv(index=False).encode("utf-8"))
except Exception as e:
    st.error(f"Could not prepare the model: {e}")
    st.info("Check that the CSV has columns: Study_Hours, Attendance, Previous_Marks, Assignment_Score, Internal_Marks, Sleep_Hours, Result.")
    st.stop()

# Sidebar navigation
st.sidebar.markdown("## 🎓 Project Menu")
page = st.sidebar.radio("Go to", ["Dashboard", "Predict Student Result", "Dataset Explorer", "Model Evaluation", "About Project"])
st.sidebar.divider()
st.sidebar.caption("Student Performance Prediction")
st.sidebar.caption("B.Tech CSE • Machine Learning")
st.sidebar.caption("Demo project — synthetic data")

if page == "Dashboard":
    st.subheader("Project overview")
    st.markdown('<p class="small-muted">Explore the dataset, compare model results, and test an example student.</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    pass_count = int((df["Result"].astype(str).str.title() == "Pass").sum())
    fail_count = int((df["Result"].astype(str).str.title() == "Fail").sum())
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total records</div><div class="metric-value">{len(df):,}</div><div class="metric-note">Available demo records</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Training records</div><div class="metric-value">{train_count:,}</div><div class="metric-note">Used to fit models</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Logistic Regression</div><div class="metric-value">{metrics["Logistic Regression"]["accuracy"]*100:.1f}%</div><div class="metric-note">Test-set accuracy</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Decision Tree</div><div class="metric-value">{metrics["Decision Tree"]["accuracy"]*100:.1f}%</div><div class="metric-note">Test-set accuracy</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Pass vs Fail distribution")
        dist = pd.DataFrame({"Result": ["Pass", "Fail"], "Students": [pass_count, fail_count]}).set_index("Result")
        st.bar_chart(dist)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Model comparison")
        comp = pd.DataFrame({
            "Model": list(metrics.keys()),
            "Accuracy (%)": [v["accuracy"]*100 for v in metrics.values()]
        }).set_index("Model")
        st.bar_chart(comp)
        st.markdown('</div>', unsafe_allow_html=True)
    st.info("Use **Predict Student Result** in the sidebar to enter student details and run both models.")
    st.warning("Educational demo only: the dataset is synthetic and predictions are not validated for real academic decisions.")

elif page == "Predict Student Result":
    st.subheader("Predict a student's result")
    st.markdown('<p class="small-muted">Enter the student information below, then click Predict. Both trained models will return a result.</p>', unsafe_allow_html=True)
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            study = st.slider("Study hours per day", 0.0, 14.0, 6.0, 0.5)
            attendance = st.slider("Attendance (%)", 0, 100, 85)
        with col2:
            previous = st.slider("Previous marks (%)", 0, 100, 75)
            assignment = st.slider("Assignment score (%)", 0, 100, 80)
        with col3:
            internal = st.slider("Internal marks (%)", 0, 100, 78)
            sleep = st.slider("Sleep hours per night", 0.0, 14.0, 7.0, 0.5)
        submitted = st.form_submit_button("✨ Predict Performance", type="primary", use_container_width=True)

    if submitted:
        sample = pd.DataFrame([{
            "Study_Hours": study, "Attendance": attendance,
            "Previous_Marks": previous, "Assignment_Score": assignment,
            "Internal_Marks": internal, "Sleep_Hours": sleep
        }])[FEATURES]
        st.markdown("### Prediction results")
        p1, p2 = st.columns(2)
        for col, name in zip([p1, p2], ["Logistic Regression", "Decision Tree"]):
            with col:
                pred = int(models[name].predict(sample)[0])
                label = "PASS" if pred == 1 else "FAIL"
                probability = float(models[name].predict_proba(sample)[0][1])
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown(f"#### {name}")
                if pred == 1:
                    st.success(f"Predicted result: {label}")
                else:
                    st.error(f"Predicted result: {label}")
                st.metric("Estimated Pass probability", f"{probability*100:.1f}%")
                st.progress(max(0, min(100, int(probability*100))))
                st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Probability is a model estimate based on synthetic training data, not a real-world guarantee.")

elif page == "Dataset Explorer":
    st.subheader("Dataset explorer")
    st.markdown('<p class="small-muted">Review the sample records and summary statistics.</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Preview")
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("#### Summary statistics")
        st.dataframe(df[FEATURES].describe().T, use_container_width=True)
    st.markdown("#### Filter records")
    result_options = ["All"] + sorted(df["Result"].astype(str).unique().tolist())
    chosen = st.selectbox("Result category", result_options)
    filtered = df if chosen == "All" else df[df["Result"].astype(str) == chosen]
    st.write(f"Showing **{len(filtered)}** records")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Download displayed data as CSV", filtered.to_csv(index=False).encode("utf-8"), "student_data_filtered.csv", "text/csv")

elif page == "Model Evaluation":
    st.subheader("Model evaluation")
    st.markdown('<p class="small-muted">Both models are evaluated on the same held-out test split.</p>', unsafe_allow_html=True)
    rows = []
    for name, m in metrics.items():
        rows.append({"Model": name, "Accuracy (%)": round(m["accuracy"]*100, 2),
                     "Fail precision": round(m["report"]["Fail"]["precision"], 3),
                     "Fail recall": round(m["report"]["Fail"]["recall"], 3),
                     "Pass precision": round(m["report"]["Pass"]["precision"], 3),
                     "Pass recall": round(m["report"]["Pass"]["recall"], 3)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    cols = st.columns(2)
    for col, (name, m) in zip(cols, metrics.items()):
        with col:
            st.markdown(f"#### {name}")
            cm = pd.DataFrame(m["matrix"], index=["Actual Fail", "Actual Pass"], columns=["Predicted Fail", "Predicted Pass"])
            st.dataframe(cm, use_container_width=True)
            st.caption("Confusion matrix: rows are actual classes; columns are predicted classes.")
    st.warning("Accuracy can be misleading when classes are imbalanced. Review precision, recall, and the confusion matrix too.")

else:
    st.subheader("About this project")
    st.markdown("""
    <div class="panel">
      <h3>Project objective</h3>
      <p>Demonstrate a basic Machine Learning workflow that classifies a student's result as Pass or Fail from study habits and assessment-related features.</p>
      <h3>Algorithms</h3>
      <ul>
        <li><b>Logistic Regression:</b> a classification model that estimates class probabilities.</li>
        <li><b>Decision Tree:</b> a model that learns rule-based splits from the training data.</li>
      </ul>
      <h3>Workflow</h3>
      <p>Dataset → feature/target separation → train/test split → model training → evaluation → interactive prediction.</p>
      <h3>Limitations</h3>
      <p>The supplied dataset is synthetic and intended for coursework demonstration. The model has not been validated for real-world use and should not be used to make decisions about actual students.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Prepared for:** B.Tech CSE Machine Learning project")
    st.markdown("**Student:** Utsav Tomar • **Roll No.:** S25CSEU2274 • **Batch:** 77")
    st.caption("You can edit these details in app.py if needed.")

st.divider()
st.caption("Student Performance Prediction System • College project demonstration")
