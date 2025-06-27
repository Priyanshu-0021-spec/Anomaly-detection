import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set style for plots
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Title
st.title("Credit Card Fraud Detection Analysis")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    return df

df = load_data()

# Sidebar
st.sidebar.header("Data Overview")
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(df.head())

# Main content
tab1, tab2, tab3 = st.tabs(["Data Overview", "Visualizations", "Feature Engineering"])

with tab1:
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Shape")
        st.write(f"Number of transactions: {df.shape[0]}")
        st.write(f"Number of features: {df.shape[1]}")
        
        st.subheader("Missing Values")
        st.write(df.isnull().sum())
    
    with col2:
        st.subheader("Class Distribution")
        fraud_percent = df['Class'].value_counts(normalize=True)[1] * 100
        st.write(f"Fraudulent transactions: {fraud_percent:.4f}%")
        
        counts = df.Class.value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts, labels=['Normal', 'Fraud'], autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)

with tab2:
    st.header("Data Visualizations")
    
    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Class', hue='Class', data=df, palette={0: 'skyblue', 1: 'red'}, legend=False, ax=ax)
    ax.set_title("Class Distribution (0 = Normal, 1 = Fraud)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    
    st.subheader("Transaction Amount Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.boxplot(x='Class', y='Amount', data=df, ax=ax[0])
    ax[0].set_title("Amount by Class")
    
    sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='skyblue', label='Normal', ax=ax[1], kde=True)
    sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', label='Fraud', ax=ax[1], kde=True)
    ax[1].set_title("Amount Distribution by Class")
    ax[1].legend()
    ax[1].set_xlim(0, 500)  # Limit x-axis for better visualization
    st.pyplot(fig)

with tab3:
    st.header("Feature Engineering")
    
    st.subheader("Feature Scaling")
    st.write("StandardScaler is applied to all features except 'Time' and 'Class'")
    
    # Drop Time column
    df_processed = df.drop(columns=['Time'])
    
    # Separate features and target
    X = df_processed.drop(columns=['Class'])
    y = df_processed['Class']
    
    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.subheader("PCA Transformation")
    n_components = st.slider("Select number of PCA components", 2, 28, 10)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    st.write(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    st.write(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")
    
    # Plot explained variance
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Explained Variance by PCA Components')
    st.pyplot(fig)
    
    if st.checkbox("Show PCA components"):
        st.write(pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)]).head())

# Key findings
st.sidebar.header("Key Findings")
st.sidebar.write("""
- Severe class imbalance (0.17% fraud)
- PCA reduced features to 10 components
- Time column dropped
- Data scaled with StandardScaler
""")

# Next steps
st.sidebar.header("Next Steps")
st.sidebar.write("""
1. Handle class imbalance
2. Select appropriate models
3. Analyze feature importance
4. Evaluate with proper metrics
5. Consider deployment needs
""")
