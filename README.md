# E-Commerce Customer Analysis using Unsupervised Learning

## Project Overview

This project applies multiple unsupervised learning techniques to analyze customer behavior in an e-commerce environment. The objective is to discover hidden customer segments, detect unusual purchasing behavior, reduce data dimensionality, and generate personalized product recommendations.

The project demonstrates practical implementation of:

* K-Means Clustering
* Density-Based Anomaly Detection
* Principal Component Analysis (PCA)
* Collaborative Filtering Recommendation System
* Streamlit Dashboard

## Dataset

The analysis uses e-commerce customer transaction and interaction data containing information such as:

* Customer purchase history
* Spending behavior
* Product interactions
* Transaction patterns

## Objectives

* Segment customers into meaningful groups
* Detect anomalous customer behavior
* Reduce feature dimensionality using PCA
* Build a recommendation system for personalized product suggestions

## Project Workflow

### 1. Data Preprocessing

* Data loading and inspection
* Missing value handling
* Duplicate removal
* Feature scaling and normalization

### 2. Customer Segmentation

* K-Means clustering applied on customer behavior features
* Optimal number of clusters selected using Elbow Method and Silhouette Score
* Cluster interpretation and visualization

### 3. Anomaly Detection

* Density estimation techniques used to identify unusual customers
* Detection of outliers based on purchasing patterns
* Visualization of normal and anomalous observations

### 4. Principal Component Analysis (PCA)

* Dimensionality reduction performed on scaled data
* Variance explained by principal components analyzed
* Two-dimensional visualization of transformed data

### 5. Recommendation System

* Collaborative filtering approach implemented
* Product recommendations generated based on user similarity
* Sample recommendations provided for multiple users

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit

## Key Insights

* Customer groups with distinct purchasing behaviors were identified.
* Anomalous customers were successfully detected.
* PCA reduced dimensionality while retaining most of the important information.
* Collaborative filtering generated personalized product recommendations.

## Streamlit Application

The project includes an interactive Streamlit dashboard for exploring clustering results, anomaly detection, PCA visualizations, and recommendation outputs.

## Future Improvements

* Advanced recommendation algorithms
* Real-time customer segmentation
* Deep learning-based anomaly detection
* Automated model deployment

## Author

Anshara Altaf
Aspiring Data Scientist
