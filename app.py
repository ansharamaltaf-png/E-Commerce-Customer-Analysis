import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import zscore

st.title(" 📊 E-Commerce Customer Analysis")

# -------------------------------
#  Task 1: Data Understanding & Preprocessing
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='latin1')

st.subheader("Task 1: Data Understanding & Preprocessing")
st.write(" - We clean and normalize the data to remove noise and make features comparable for unsupervised learning.")
st.write(df.head())

# Drop missing customer IDs
df = df.dropna(subset=["CustomerID"])

# Remove duplicates
df = df.drop_duplicates()

# Create Total Price
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Aggregate customer-level data
customer_data = df.groupby("CustomerID").agg({
    "Quantity": "sum",
    "TotalPrice": "sum",
    "InvoiceNo": "nunique"
}).reset_index()

customer_data.columns = ["CustomerID", "TotalQuantity", "TotalSpent", "NumTransactions"]

st.subheader("Customer Aggregated Data")
st.write(customer_data.head())

# Scale features
features = ["TotalQuantity", "TotalSpent", "NumTransactions"]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[features])

# -------------------------------
# Task 2: Customer Segmentation using K-Means
# -------------------------------

st.subheader("Task 2: Customer Segmentation using K-Means")
st.write(" - K-Means clusters customers into groups based on similar purchase behavior to identify distinct segments.")

wcss = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

fig1 = plt.figure()
plt.plot(range(2, 10), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
st.pyplot(fig1)

kmeans = KMeans(n_clusters=4, random_state=42)
customer_data["Cluster"] = kmeans.fit_predict(scaled_data)

st.write("Silhouette Score:", silhouette_score(scaled_data, customer_data["Cluster"]))

fig2 = plt.figure()
sns.scatterplot(
    x=customer_data["TotalSpent"],
    y=customer_data["TotalQuantity"],
    hue=customer_data["Cluster"],
    palette="viridis"
)
st.pyplot(fig2)

# -------------------------------
# Task 3: Density Estimation & Anomaly Detection
# -------------------------------

st.subheader("Task 3: Density Estimation & Anomaly Detection")
st.write(" - We detect unusual customers whose purchasing patterns differ significantly from the majority using density-based methods.")

z_scores = np.abs(zscore(customer_data[features]))
customer_data["Z_score"] = np.sqrt((z_scores**2).sum(axis=1))

threshold = 3
customer_data["Anomaly"] = customer_data["Z_score"] > threshold

fig3 = plt.figure()
sns.scatterplot(
    x=customer_data["TotalSpent"],
    y=customer_data["TotalQuantity"],
    hue=customer_data["Anomaly"]
)
st.pyplot(fig3)

# -------------------------------
# Task 4: Dimensionality Reduction using PCA
# -------------------------------

st.subheader("Task 4: Dimensionality Reduction using PCA")
st.write(" - PCA reduces feature dimensions while preserving most variance, making patterns easier to visualize and interpret.")


pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

fig4 = plt.figure()
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=customer_data["Cluster"])
plt.xlabel("PC1")
plt.ylabel("PC2")
st.pyplot(fig4)

# ===============================
# Task 5: Recommendation System using Collaborative Filtering
# ===============================

st.subheader("Task 5: Recommendation System using Collaborative Filtering")
st.write(" - Collaborative filtering recommends products to a customer based on the preferences of similar users.")

# Create user-item matrix
user_item_matrix = df.pivot_table(
    index="CustomerID",
    columns="StockCode",
    values="Quantity",
    fill_value=0
)
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(user_item_matrix)
customer_id = st.selectbox(
    "Select Customer ID for Recommendation",
    user_item_matrix.index
)
if st.button("Generate Recommendations"):

    user_index = user_item_matrix.index.get_loc(customer_id)

    similarity_scores = similarity_matrix[user_index]

    similar_users = similarity_scores.argsort()[::-1][1:6]

    recommended_products = (
        user_item_matrix.iloc[similar_users]
        .mean()
        .sort_values(ascending=False)
    )

    st.write("Top 5 Recommended Products:")
    st.write(recommended_products.head(5))

st.subheader("Task 6: Analysis & Reflection")

st.markdown("**How unsupervised learning helped uncover hidden patterns:**")
st.write("""
- **Clustering (K-Means):** Identified customer segments like high spenders, frequent buyers, and one-time buyers.  
- **Anomaly Detection:** Highlighted unusual customer behavior or extreme purchases.  
- **PCA:** Reduced dimensions, making patterns and clusters easier to visualize.  
- **Collaborative Filtering:** Recommended products based on similar users' behavior.
""")

st.markdown("**Comparison of techniques:**")
comparison_data = {
    "Method": ["Clustering", "Anomaly Detection", "PCA", "Collaborative Filtering"],
    "Usefulness": [
        "Groups customers with similar behaviors for targeted marketing.",
        "Detects unusual transactions or outliers for fraud detection.",
        "Reduces data dimensionality to visualize and interpret patterns.",
        "Suggests products to customers based on others with similar preferences."
    ]
}
comparison_df = pd.DataFrame(comparison_data)
st.table(comparison_df)

st.markdown("**Real-world applications:**")
st.write("""
- **E-commerce:** Personalized product recommendations for customers.  
- **Banking & Fintech:** Detect fraudulent or unusual transactions.  
- **Marketing & Retail:** Customer segmentation for targeted campaigns and inventory optimization.  
- **Data Analysis:** Visualizing patterns in large datasets for decision making.
""")
