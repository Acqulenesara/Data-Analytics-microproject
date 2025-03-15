import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from flask import Flask, render_template, request
import io
import base64

app = Flask(__name__)

# Set Pandas display options
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Avoid line breaks
pd.set_option('display.float_format', '{:.2f}'.format)  # Optional: format floats

# Data Preprocessing
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    return df

# RFM Analysis
def perform_rfm_analysis(df):
    today = df['Date'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'Date': lambda x: (today - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm_log = rfm.apply(lambda x: np.log1p(x))
    return rfm, rfm_log

# Calculate RFM Scores
def calculate_rfm_scores(rfm):
    rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=False, duplicates='drop') + 1
    rfm['F_score'] = pd.qcut(rfm['Frequency'], 5, labels=False, duplicates='drop')
    rfm['M_score'] = pd.qcut(rfm['Monetary'], 5, labels=False, duplicates='drop')
    rfm['RFM_Score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
    return rfm

# Clustering
def perform_clustering(rfm_log, n_clusters=4):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm_log['KMeans_Cluster'] = kmeans.fit_predict(rfm_scaled)
    kmeans_silhouette = silhouette_score(rfm_scaled, rfm_log['KMeans_Cluster'])

    birch = Birch(n_clusters=n_clusters)
    rfm_log['BIRCH_Cluster'] = birch.fit_predict(rfm_scaled)
    birch_silhouette = silhouette_score(rfm_scaled, rfm_log['BIRCH_Cluster'])

    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    rfm_log['Agglomerative_Cluster'] = agglomerative.fit_predict(rfm_scaled)
    agglomerative_silhouette = silhouette_score(rfm_scaled, rfm_log['Agglomerative_Cluster'])

    print("K-means Silhouette Score:", kmeans_silhouette)
    print("BIRCH Silhouette Score:", birch_silhouette)
    print("Agglomerative Silhouette Score:", agglomerative_silhouette)

    return rfm_scaled, rfm_log, (kmeans_silhouette, birch_silhouette, agglomerative_silhouette)

# Customer Segmentation
def segment_customers(rfm_log):
    def assign_segment(row):
        if row['R_score'] >= 4 and row['F_score'] >= 4 and row['M_score'] >= 4:
            return 'Best Customers'
        elif row['R_score'] <= 2 and row['F_score'] <= 2 and row['M_score'] <= 2:
            return 'Lost Customers'
        elif row['R_score'] >= 3 and row['F_score'] <= 2 and row['M_score'] <= 2:
            return 'New Customers'
        elif row['R_score'] <= 2 and row['F_score'] >= 3 and row['M_score'] >= 3:
            return 'Loyal Customers'
        else:
            return 'Average Customers'

    rfm_log['Segment'] = rfm_log.apply(assign_segment, axis=1)
    return rfm_log

# Visualization
def generate_plots(rfm_scaled, rfm_log):
    pca = PCA(n_components=2)
    rfm_pca = pca.fit_transform(rfm_scaled)

    plt.figure(figsize=(20, 5))

    plt.subplot(131)
    plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=rfm_log['KMeans_Cluster'], cmap='viridis')
    plt.title('K-means Clustering')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(label='Cluster ID')

    plt.subplot(132)
    plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=rfm_log['BIRCH_Cluster'], cmap='viridis')
    plt.title('BIRCH Clustering')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(label='Cluster ID')

    plt.subplot(133)
    plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=rfm_log['Agglomerative_Cluster'], cmap='viridis')
    plt.title('Agglomerative Clustering')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(label='Cluster ID')

    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the bytes buffer in base64
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    plt.close()
    
    return plot_url

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            df = preprocess_data(df)

            # Perform RFM analysis
            rfm, rfm_log = perform_rfm_analysis(df)
            rfm = calculate_rfm_scores(rfm)
            rfm_scaled, rfm_log, silhouette_scores = perform_clustering(rfm_log, n_clusters=4)
            rfm_log = rfm_log.join(rfm[['R_score', 'F_score', 'M_score', 'RFM_Score']])
            rfm_log = segment_customers(rfm_log)

            # Convert DataFrame to HTML with only the specified columns
            table_html = rfm_log[['KMeans_Cluster', 'Segment', 'Recency', 'Frequency', 'Monetary', 'RFM_Score']].to_html()

            # Generate plots
            plot_url = generate_plots(rfm_scaled, rfm_log)

            # Display silhouette scores
            kmeans_score, birch_score, agglomerative_score = silhouette_scores

            return render_template('index.html', table=table_html, plot_url=plot_url,
                                   kmeans_score=kmeans_score, birch_score=birch_score,
                                   agglomerative_score=agglomerative_score)

if __name__ == '__main__':
    app.run(debug=True)
