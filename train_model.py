"""
train_model.py
----------------
Reproduces the K-Means customer segmentation pipeline from the notebook:
    Gender label-encoding -> StandardScaler -> KMeans (k=5)

Saves the fitted encoder, scaler, and KMeans model so the Streamlit app can
assign a NEW customer to one of the 5 existing segments instantly, without
retraining.

Run:
    python train_model.py

Outputs (in model/):
    label_encoder.pkl   -> fitted LabelEncoder for Gender
    scaler.pkl           -> fitted StandardScaler
    kmeans_model.pkl      -> fitted KMeans (n_clusters=5)
    clustered_data.pkl    -> original data + assigned Cluster column
    cluster_profiles.pkl  -> human-readable name/description per cluster
"""

import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_PATH = "Mall_Customers.csv"
MODEL_DIR = "model"
N_CLUSTERS = 5


def label_cluster(row):
    """Give each cluster centroid a human-readable business label based on
    income and spending score, matching the notebook's business insights."""
    income, spending, gender = row["Annual Income (k$)"], row["Spending Score (1-100)"], row["Gender"]
    gender_tag = "Mostly Male" if gender >= 0.5 else "Mostly Female"
    if income >= 70 and spending >= 60:
        return "High Income, High Spending (Premium Target)"
    if income >= 70 and spending < 60:
        return "High Income, Low Spending (Cautious Spenders)"
    if income < 45 and spending >= 60:
        return "Low Income, High Spending (Impulsive)"
    if income < 45 and spending < 45:
        return "Low Income, Low Spending (Budget Conscious)"
    return f"Average Income, Average Spending, {gender_tag} (Standard)"


def main():
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    X = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters

    score = silhouette_score(X_scaled, clusters)
    print("Silhouette Score:", round(score, 3))

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers = pd.DataFrame(
        centers, columns=["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
    )
    cluster_centers["Cluster"] = range(N_CLUSTERS)
    cluster_centers["Label"] = cluster_centers.apply(label_cluster, axis=1)
    print("\nCluster profiles:\n", cluster_centers)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    joblib.dump(df, os.path.join(MODEL_DIR, "clustered_data.pkl"))
    joblib.dump(cluster_centers, os.path.join(MODEL_DIR, "cluster_profiles.pkl"))

    print("\nSaved model files to model/")


if __name__ == "__main__":
    main()
