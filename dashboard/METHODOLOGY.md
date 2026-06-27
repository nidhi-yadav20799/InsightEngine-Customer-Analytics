# InsightEngine — Customer Analytics Dashboard
## Methodology & Findings Document

**Project:** E-Commerce Customer Analytics Dashboard
**Dataset:** Online Retail II (UCI / Kaggle) — Year 2009-2010
**Tech Stack:** Python, Pandas, Scikit-learn, Streamlit, Matplotlib/Seaborn, SciPy

---

## 1. Data Cleaning

- Removed rows with missing `Customer ID` — guest/unidentified transactions can't be used for customer-level analysis (RFM, segmentation, churn, recommendations all require a known customer).
- Removed rows with `Quantity <= 0` or `Price <= 0` — these represent cancellations, returns, and data entry errors, not genuine purchases. Including them would distort revenue and quantity metrics.
- Converted `InvoiceDate` to proper datetime format for time-based analysis.
- Created a derived feature `TotalAmount = Quantity × Price` for revenue calculations.

## 2. Exploratory Data Analysis (EDA)

- **Revenue Over Time:** Daily revenue trend reveals seasonality and spikes — useful for identifying peak sales periods (e.g. pre-holiday demand).
- **Order Value Distribution:** Most orders are low-to-mid value (right-skewed distribution), typical of B2C retail. A small number of large bulk orders are present, consistent with wholesale/B2B-style buyers in the dataset.
- **Top Products by Revenue vs. Quantity Variability:** Revenue leaders aren't always the highest-quantity items — distinguishing "frequently bought in bulk" items from "high unit-price" items.
- **Revenue & Customers by Country:** Revenue is heavily concentrated in the United Kingdom, consistent with this being a UK-based retailer; other countries contribute a smaller tail.

## 3. Feature Engineering — RFM

RFM (Recency, Frequency, Monetary) was computed per customer:
- **Recency:** Days since the customer's last purchase (snapshot date = last invoice date + 1 day).
- **Frequency:** Count of unique invoices (orders) per customer.
- **Monetary:** Total spend per customer.

These three features capture engagement, loyalty, and value — the standard basis for customer segmentation in retail analytics.

## 4. Customer Segmentation

- **Method:** K-Means clustering (k=4) on standardized RFM features (`StandardScaler` applied before clustering, since K-Means is distance-based and Monetary/Frequency/Recency are on very different scales).
- **Segment naming:** Clusters are ranked and labeled by average Monetary value (high → low) rather than fixed cluster IDs, since K-Means cluster labels (0,1,2,3) are arbitrary and can shift between runs.
- **Resulting segments:**
  - **High Value Customers** — highest spend, most frequent, most recent
  - **Loyal Customers** — frequent buyers with solid spend
  - **Regular Customers** — average activity across all three dimensions
  - **At Risk Customers** — high recency (long time since last purchase), low frequency — candidates for re-engagement campaigns

## 5. Churn Prediction

- **Definition of churn:** A customer is labeled "churned" if Recency > 90 days (no purchase in the last 3 months) — a standard threshold for retail churn given typical repeat-purchase cycles in this dataset.
- **Model:** Logistic Regression on Recency, Frequency, Monetary — chosen for interpretability (coefficients directly show which factor drives churn risk) over a black-box model, which matters for business stakeholders who need to act on the output.
- **Output:** A churn probability score per customer (not just a binary label), so the business can prioritize outreach to the highest-risk, highest-value customers first rather than treating all "at risk" customers equally.

## 6. Recommendation Engine

- **Method:** User-based collaborative filtering using cosine similarity on a customer–product purchase matrix (rows = customers, columns = products, values = total quantity purchased).
- **Logic:** For a given customer, find the 10 most similar customers (by purchase pattern), aggregate what those similar customers bought, exclude products the target customer already purchased, and recommend the top 5 remaining products by aggregate score.
- **Limitation:** Cold-start problem — new customers with very few purchases have less reliable similarity scores. This is a known limitation of collaborative filtering and would be addressed with content-based filtering (product attributes) in a production system.

## 7. Statistical Testing (A/B Simulation)

- **Method:** Welch's independent two-sample t-test (`scipy.stats.ttest_ind`, `equal_var=False`) comparing average order value between two customer groups, segmented by country.
- **Why Welch's t-test:** Country-level order value groups are unlikely to have equal variances or equal sample sizes — Welch's test is robust to both, unlike the standard Student's t-test.
- **Hypotheses:**
  - H₀: Mean order value is equal between Group A and Group B
  - H₁: Mean order value differs between Group A and Group B
- **Significance level:** α = 0.05. p-value < 0.05 → statistically significant difference; otherwise, fail to reject H₀.

## 8. Key Business Insights

1. Revenue is heavily UK-concentrated — international expansion or region-specific marketing could unlock growth in underrepresented countries.
2. "At Risk Customers" represent a meaningful share of the customer base with high historical Monetary value but rising Recency — a targeted win-back campaign (discount/email) is the highest-leverage action here.
3. High Value Customers, while fewer in number, contribute disproportionately to revenue (Pareto pattern) — retention efforts should prioritize this segment over acquisition spend.
4. Bulk-quantity orders for specific SKUs (seen in the box plot analysis) suggest a wholesale/reseller customer subset distinct from typical retail buyers — this could justify a separate B2B pricing tier.
5. Country-level A/B testing shows [statistically significant / not significant — depends on countries selected in the live dashboard] differences in average order value, suggesting [region-specific pricing strategies may be justified / order value behaviour is consistent across regions].

## 9. Limitations & Future Work

- Single-year dataset (2009-2010) — seasonal patterns are visible but year-over-year trend analysis isn't possible.
- Churn threshold (90 days) is a heuristic, not derived from survival analysis — a Cox proportional hazards model would give a more rigorous definition.
- Recommendation engine doesn't account for product category/price tier — a hybrid (collaborative + content-based) approach would likely outperform pure collaborative filtering.