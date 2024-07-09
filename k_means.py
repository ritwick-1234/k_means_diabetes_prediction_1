import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split


st.sidebar.header('K-Means Clustering Hyperparameters')
n_clusters = st.sidebar.slider('Number of clusters (n_clusters)', min_value=1, max_value=300, value=8)
init = st.sidebar.selectbox('Initialization method (init)', options=['k-means++', 'random'], index=0)
n_init = st.sidebar.selectbox('Number of initializations (n_init)', options=['auto', 1, 10, 100], index=0)
max_iter = st.sidebar.number_input('Maximum number of iterations (max_iter)', min_value=1, max_value=1000, value=300)
algorithm = st.sidebar.radio('Algorithm', options=['lloyd', 'elkan'], index=0)


df = pd.read_csv("diabetes_1234_project.csv")


def get_mapped_labels(true_labels, pred_labels, n_clusters):
    # Create a cost matrix
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost_matrix[i, j] = np.sum((true_labels == i) & (pred_labels == j))
    
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    return np.array([col_ind[label] for label in pred_labels])


features = df.drop(columns=['Outcome'])
true_labels = df['Outcome']




X_train, X_test, y_train, y_test = train_test_split(features, true_labels, test_size=0.2, random_state=42)


if st.sidebar.button('Submit'):

    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, algorithm=algorithm)
    kmeans.fit(X_train)
    labels_train = kmeans.labels_
    labels_test = kmeans.predict(X_test)
    

    labels_train_mapped = get_mapped_labels(y_train, labels_train, n_clusters)
    labels_test_mapped = get_mapped_labels(y_test, labels_test, n_clusters)
    
    
    accuracy_train = accuracy_score(y_train, labels_train_mapped)
    accuracy_test = accuracy_score(y_test, labels_test_mapped)
    
  
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    n_iter = kmeans.n_iter_
    n_features_in = kmeans.n_features_in_
    
    
    st.write("### K-Means Clustering Results")
    st.write(f"**Cluster Centers:** \n{cluster_centers}")
    st.write(f"**Labels:** \n{labels}")
    st.write(f"**Inertia:** {inertia}")
    st.write(f"**Number of Iterations:** {n_iter}")
    st.write(f"**Number of Features:** {n_features_in}")
    st.write(f"**Training Accuracy:** {accuracy_train*100:.2f}")
    st.write(f"**Test Accuracy:** {accuracy_test*100:.2f}")
