#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:52:00 2025

@author: megha
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

def main():
    st.title("Order Clustering App")
    st.write("""
    **Instructions**:
    1. Upload a CSV or Excel file with columns: `order`, `sku`, `quantity`.
    2. We will cluster similar orders based on their SKU quantities.
    3. You'll see which orders are grouped together, and a 2D plot showing the clusters.
    """)

    # -------------------------------------
    # 1. File Uploader
    # -------------------------------------
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    # If no file is uploaded yet, just show instructions
    if uploaded_file is None:
        st.stop()

    # -------------------------------------
    # 2. Read the Data
    # -------------------------------------
    # We'll attempt to read as CSV first; if that fails, try Excel
    try:
        df = pd.read_csv(uploaded_file)
    except:
        df = pd.read_excel(uploaded_file)

    # Validate required columns
    required_cols = {"order", "sku", "quantity"}
    if not required_cols.issubset(df.columns):
        st.error(f"Uploaded file must contain at least these columns: {required_cols}")
        st.stop()

    # Show a sample of the data
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # -------------------------------------
    # 3. Pivot / Reshape the Data
    #    Rows = order, Columns = sku, Values = quantity
    # -------------------------------------
    # If an order-sku is missing in the CSV, quantity is 0
    pivot_df = df.pivot_table(
        index="order",
        columns="sku",
        values="quantity",
        aggfunc="sum",
        fill_value=0
    )

    st.subheader("Pivoted Data (Orders x SKUs)")
    st.dataframe(pivot_df.head())

    # Convert pivot table to a NumPy array
    order_matrix = pivot_df.values
    order_ids = pivot_df.index.to_list()  # We'll use this to identify orders in results

    # -------------------------------------
    # 4. Dimensionality Reduction (TruncatedSVD)
    #    - Helps if you have many SKUs (high-dimensional).
    #    - We'll reduce to 2D so we can visualize easily.
    # -------------------------------------
    n_components = 2
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    order_matrix_reduced = svd.fit_transform(order_matrix)

    # -------------------------------------
    # 5. Clustering (Agglomerative, Cosine Distance)
    #    - We'll ask user for number of clusters, or use a default (e.g. 3).
    # -------------------------------------
    st.subheader("Clustering Parameters")
    k = st.number_input("Number of Clusters (k)", min_value=2, max_value=10, value=3, step=1)

    # Compute a distance matrix using cosine distance
    distance_matrix = pairwise_distances(order_matrix_reduced, metric="cosine")

    # Perform Agglomerative Clustering
    agg = AgglomerativeClustering(
        n_clusters=k,
        affinity="precomputed",
        linkage="average"
    )
    labels = agg.fit_predict(distance_matrix)

    # -------------------------------------
    # 6. Show Cluster Assignments
    # -------------------------------------
    # Create a DataFrame that maps each order to its cluster
    cluster_df = pd.DataFrame({
        "order": order_ids,
        "cluster": labels
    }).sort_values(by="cluster")

    st.subheader("Cluster Assignments")
    st.dataframe(cluster_df)

    # We can also group them for printing
    st.markdown("**Orders grouped by cluster:**")
    for cluster_label in sorted(cluster_df["cluster"].unique()):
        orders_in_cluster = cluster_df[cluster_df["cluster"] == cluster_label]["order"].tolist()
        st.write(f"- **Cluster {cluster_label}**: {orders_in_cluster}")

    # -------------------------------------
    # 7. Plot in 2D (SVD Components) with Cluster Coloring
    # -------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        order_matrix_reduced[:, 0],
        order_matrix_reduced[:, 1],
        c=labels,
        cmap="viridis",
        s=100
    )
    ax.set_title("Orders (2D SVD) with Agglomerative Clusters (Cosine)")
    ax.set_xlabel("SVD Component 1")
    ax.set_ylabel("SVD Component 2")

    # Annotate points with the order ID
    for i, order_id in enumerate(order_ids):
        ax.text(
            order_matrix_reduced[i, 0],
            order_matrix_reduced[i, 1],
            f" {order_id}",
            fontsize=8,
            verticalalignment="bottom"
        )

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster Label")

    st.pyplot(fig)

if __name__ == "__main__":
    main()
