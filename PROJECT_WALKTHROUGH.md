# Project Walkthrough: E-Commerce Customer Analysis & RFM Segmentation

## 1. High-Level Architecture
This project processes raw transactional records through six structured processing stages:
```
Raw Transactions (.csv.gz) 
  ──> Quality Inspection (Nulls, Duplicates, Return Codes)
  ──> Data Cleansing (Filters, Timestamp Parsing, TotalPrice)
  ──> Macro Analysis (Monthly Revenue, Top SKUs, Geography)
  ──> Micro RFM Segmentation (Recency, Frequency, Monetary Quartiles)
  ──> Publication Visualizations (.png)
  ──> KPI Metrics Export (.json)
```

---

## 2. Core Functions & Design Decisions Explained

### `clean_retail_data(df: pd.DataFrame)`
- **Why filter `InvoiceNo.str.startswith('C')`?**: Retail invoice systems prefix invoices with 'C' when an order is cancelled or refunded. If kept without adjustment, they distort sales totals.
- **Why compute `TotalPrice = Quantity * UnitPrice`?**: Individual lines store unit costs; multiplying by quantity gives the gross line-item contribution.

### `perform_rfm_segmentation(df: pd.DataFrame)`
- **Anchor Date Calculation**:
  ```python
  reference_date = cust_df["InvoiceDate"].max() + pd.Timedelta(days=1)
  ```
  We add 1 day to the max recorded date so that the most recent buyers have a positive non-zero recency (e.g., 1 day).
- **Quartile Scoring with `rank(method='first')`**:
  Ties in frequency (e.g., hundreds of customers with exactly 1 purchase) can cause bin edge collisions in `pd.qcut()`. Applying `.rank(method="first")` breaks ties stably and ensures exactly 4 balanced quartiles.

---

## 3. Three Improvements You Can Personally Implement

1. **Market Basket Association Analysis (Apriori / FP-Growth)**:
   - Identify pairs or sets of items frequently purchased together in the same `InvoiceNo`.
   - Use the `mlxtend` Python library to calculate Support, Confidence, and Lift.
2. **Customer Cohort Retention Heatmap**:
   - Group customers by the month of their first purchase (cohort month).
   - Track what percentage of each cohort returned to make a purchase in months 1 through 12.
3. **Streamlit Interactive Web App**:
   - Build a lightweight UI allowing a user to adjust the RFM score thresholds using sliders and download targeted customer lists as CSV.

---

## 4. Hands-On Practice Exercises
- **Exercise 1**: Modify `clean_retail_data` to calculate the percentage of total transactions originating from wholesale buyers (orders with `Quantity > 50`).
- **Exercise 2**: Write a function that returns the top 5 customers who haven't made a purchase in over 180 days but spent over £1,000 historically (High-Value Churned).
- **Exercise 3**: Plot a boxplot comparing the order values (`TotalPrice`) between the United Kingdom and Germany.
