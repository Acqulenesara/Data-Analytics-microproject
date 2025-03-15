# **Customer Segmentation Using RFM Analysis and Clustering**  

## **Overview**  
This project is a **Flask-based web application** that performs **RFM (Recency, Frequency, Monetary) analysis** and applies **clustering algorithms (K-Means, BIRCH, and Agglomerative Clustering)** to segment customers based on their purchasing behavior.  

The application allows users to **upload transaction data (CSV format)**, processes the data, performs clustering, and visualizes the customer segments using **PCA (Principal Component Analysis)** plots.  

---

## **Features**  
✅ **Data Preprocessing** – Cleans the data, removes invalid entries, and calculates total purchase values.  
✅ **RFM Analysis** – Computes **Recency, Frequency, and Monetary values** for each customer.  
✅ **Customer Segmentation** – Assigns customers to clusters using **K-Means, BIRCH, and Agglomerative Clustering**.  
✅ **Silhouette Score Calculation** – Evaluates clustering performance.  
✅ **Interactive Visualization** – Displays PCA-based cluster plots and tabular RFM scores.  
✅ **Flask Web Interface** – User-friendly UI for uploading files and viewing results.  

---

## **Technologies Used**  
- **Python** (pandas, numpy, matplotlib, seaborn)  
- **Machine Learning** (scikit-learn, K-Means, BIRCH, Agglomerative Clustering, PCA)  
- **Flask** (Web framework for UI)  
- **HTML, CSS, Bootstrap** (Frontend)  

---

## **Usage**  

1️⃣ **Upload CSV File** – The dataset should contain at least:  
   - `Date` (Transaction date)  
   - `Quantity` (Number of items bought)  
   - `UnitPrice` (Price per item)  
   - `CustomerID` (Unique identifier for customers)  

2️⃣ **Processing & Analysis** – The app performs **RFM analysis** and applies clustering algorithms.  

3️⃣ **Visualization & Insights** – Results are displayed in tabular form along with **interactive cluster plots**.  

---

## **Example CSV Format**  
| InvoiceNo | Date       | CustomerID | Quantity | UnitPrice |
|-----------|-----------|------------|----------|-----------|
| 12345     | 2024-03-01 | 1001       | 3        | 5.99      |
| 12346     | 2024-03-05 | 1002       | 1        | 15.49     |

---

## **Future Enhancements**  
🔹 **Deploy to Cloud** (e.g., AWS, Google Cloud)  
🔹 **Automated Reporting & Insights**  
🔹 **More Clustering Algorithms**  
