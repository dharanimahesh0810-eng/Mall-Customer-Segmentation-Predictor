"""
app.py
------
Streamlit web interface for Mall Customer Segmentation (K-Means).

Loads the fitted LabelEncoder, StandardScaler, and KMeans model built by
train_model.py. The user enters a new customer's details and the app
instantly assigns them to one of the 5 existing segments, shows the
segment's business profile, and plots where they land relative to all
other customers.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="centered",
)

MODEL_DIR = "model"
PATHS = {
    "encoder": os.path.join(MODEL_DIR, "label_encoder.pkl"),
    "scaler": os.path.join(MODEL_DIR, "scaler.pkl"),
    "kmeans": os.path.join(MODEL_DIR, "kmeans_model.pkl"),
    "data": os.path.join(MODEL_DIR, "clustered_data.pkl"),
    "profiles": os.path.join(MODEL_DIR, "cluster_profiles.pkl"),
}


@st.cache_resource
def load_artifacts():
    if not all(os.path.exists(p) for p in PATHS.values()):
        return None
    return {k: joblib.load(v) for k, v in PATHS.items()}


artifacts = load_artifacts()

st.title("🛍️ Mall Customer Segmentation")
st.write(
    "Enter a customer's details to see which segment they belong to, "
    "based on a K-Means clustering model trained on mall customer data "
    "(Gender, Age, Annual Income, Spending Score)."
)

if artifacts is None:
    st.error(
        "Model files not found. Run `python train_model.py` first "
        "(make sure `Mall_Customers.csv` is in this folder), then reload this app."
    )
    st.stop()

encoder = artifacts["encoder"]
scaler = artifacts["scaler"]
kmeans = artifacts["kmeans"]
df = artifacts["data"]
profiles = artifacts["profiles"]

st.divider()

# ---------- Input form ----------
with st.form("customer_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.radio("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=15, max_value=100, value=30)
    with col2:
        income = st.number_input("Annual Income (k$)", min_value=0, max_value=300, value=60)
        spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)

    submitted = st.form_submit_button("Find Segment", use_container_width=True)

if submitted:
    gender_encoded = encoder.transform([gender])[0]
    input_df = pd.DataFrame([{
        "Gender": gender_encoded,
        "Age": age,
        "Annual Income (k$)": income,
        "Spending Score (1-100)": spending,
    }])

    input_scaled = scaler.transform(input_df)
    cluster = kmeans.predict(input_scaled)[0]
    profile = profiles[profiles["Cluster"] == cluster].iloc[0]

    st.divider()
    st.subheader("Result")
    st.success(f"This customer belongs to **Segment {cluster}**")
    st.markdown(f"**Profile:** {profile['Label']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Segment Avg. Age", f"{profile['Age']:.0f}")
    col2.metric("Segment Avg. Income", f"${profile['Annual Income (k$)']:.0f}k")
    col3.metric("Segment Avg. Spending", f"{profile['Spending Score (1-100)']:.0f}/100")

    st.markdown("**All segment profiles:**")
    st.dataframe(
        profiles[["Cluster", "Age", "Annual Income (k$)", "Spending Score (1-100)", "Label"]]
        .rename(columns={"Cluster": "Segment"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Where this customer falls among all mall customers:**")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="Annual Income (k$)", y="Spending Score (1-100)",
        hue="Cluster", palette="Set2", s=80, ax=ax, legend="full",
    )
    ax.scatter(
        income, spending, color="black", marker="*", s=400,
        label="This Customer", edgecolor="white", linewidth=1.5,
    )
    ax.set_title("Mall Customer Segments (K-Means)")
    ax.legend()
    st.pyplot(fig)

st.divider()
st.caption(
    "Model: K-Means clustering (k=5) on Gender, Age, Annual Income, and "
    "Spending Score, trained on the standard 200-row Mall Customers dataset."
)
