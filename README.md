#  InsightEngine – E-Commerce Customer Analytics Dashboard

An end-to-end Data Science project that analyzes customer purchasing behavior using the Online Retail II dataset. The project covers the complete analytics pipeline—from data cleaning and exploratory analysis to customer segmentation, churn prediction, and product recommendation—deployed as an interactive Streamlit dashboard.

---

##  Live Demo

🔗 https://insightengine-customer-analytics.streamlit.app

---

## 📂 Project Structure

```
InsightEngine-Customer-Analytics/
│
├── dashboard/
│   ├── app.py
│   ├── requirements.txt
│   └── METHODOLOGY.md
│
├── data/
│   └── online_retail_II.xlsx
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_segmentation.ipynb
│   ├── 04_churn.ipynb
│   └── 05_recommender.ipynb
│
└── README.md
```

---

#  Project Overview

This project performs comprehensive customer analytics using transactional retail data.

The workflow includes:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- RFM Customer Segmentation
- Customer Churn Prediction
- Product Recommendation System
- Interactive Streamlit Dashboard

---

#  Dataset

**Dataset:** Online Retail II

- UK-based online retail transactions
- December 2009 – December 2010
- 500K+ transactions
- Customer-level purchase history

Source:
https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

---

#  Tech Stack

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn
- Plotly

### Machine Learning

- Scikit-learn

### Dashboard

- Streamlit

### Statistics

- SciPy

---

#  Project Workflow

##  Data Cleaning

- Missing value handling
- Duplicate removal
- Invalid quantity removal
- Invalid price removal
- Date formatting
- Revenue calculation

---

##  Exploratory Data Analysis

Performed multiple visual analyses including:

- Revenue over time
- Sales distribution
- Country-wise sales
- Top products
- Order value distribution
- Customer purchasing trends
- Correlation analysis

---

##  Feature Engineering

Created customer-level features including:

- Recency
- Frequency
- Monetary Value
- Customer Lifetime Value (CLV)
- Purchase Frequency
- Basket Size

---

##  Customer Segmentation

Applied RFM Analysis followed by:

- K-Means Clustering
- Elbow Method
- Silhouette Score
- Segment Profiling

Customer segments include:

- High Value Customers
- Loyal Customers
- Regular Customers
- At Risk Customers

---

##  Churn Prediction

Customers were labeled as churned using a 90-day inactivity rule.

Models implemented:

- Logistic Regression
- Random Forest
- Tuned Random Forest

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Feature Importance

---

##  Recommendation Engine

Implemented a recommendation system using:

- User-Item Matrix
- Collaborative Filtering
- Cosine Similarity

Top-N product recommendations are generated for each customer.

---

#  Streamlit Dashboard Features

The dashboard includes:

- Interactive Filters
- Revenue KPIs
- EDA Visualizations
- Customer Segmentation
- Churn Prediction
- Product Recommendations
- Business Insights

---

#  Dashboard Preview

### Overview

- Revenue
- Customers
- Orders
- Average Order Value

### EDA

- Revenue Trends
- Sales Distribution
- Country Analysis
- Product Analysis

### Customer Segmentation

- Cluster Distribution
- Segment Profiles
- RFM Analysis

### Churn Prediction

- Prediction Model
- Customer Risk
- Model Performance

### Recommendation System

- Customer Recommendations
- Similar Customer Products

---

# Model Performance

### Logistic Regression

| Metric | Score |
|---------|-------|
| Accuracy | 72.75% |
| Precision | 60.91% |
| Recall | 50.68% |
| F1 Score | 55.33% |
| ROC-AUC | 76.96% |

---

### Tuned Random Forest

| Metric | Score |
|---------|-------|
| Accuracy | 72.29% |
| Precision | 58.66% |
| Recall | 56.85% |
| F1 Score | 57.74% |
| ROC-AUC | 76.94% |

---

#  Key Business Insights

- Revenue is highly concentrated among a small group of customers.
- High-value customers contribute significantly to overall sales.
- Approximately one-third of customers exhibit churn behavior.
- RFM segmentation effectively identifies customer value groups.
- Recommendation systems can improve customer retention and cross-selling opportunities.

---

#  Installation

Clone the repository

```bash
git clone https://github.com/nidhi-yadav20799/InsightEngine-Customer-Analytics.git
```

Install dependencies

```bash
pip install -r dashboard/requirements.txt
```

Run Streamlit

```bash
streamlit run dashboard/app.py
```

---

#  Project Deliverables

- Complete Data Cleaning Pipeline
- Exploratory Data Analysis
- Customer Segmentation
- Churn Prediction Models
- Recommendation Engine
- Streamlit Dashboard
- Business Insights
- Methodology Documentation

---

#  Author

**Nidhi Yadav**


GitHub:
https://github.com/nidhi-yadav20799


---

#  If you found this project useful, consider giving it a star.
