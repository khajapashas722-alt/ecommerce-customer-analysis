"""
E-Commerce Customer Analysis & RFM Segmentation
Author: Shaik Khaja Pasha
Dataset: UCI Online Retail (UK-based Gift & Homeware Retailer)
Technology Stack: Python 3, Pandas, NumPy, Matplotlib, Seaborn
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Matplotlib style for publication-ready figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset supporting both compressed (.gz) and raw .csv files."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    print(f"[1/7] Loading dataset from: {filepath}...")
    df = pd.read_csv(filepath, compression="gzip" if filepath.endswith(".gz") else None, encoding="unicode_escape")
    print(f"      Initial records loaded: {len(df):,} rows, {len(df.columns)} columns.")
    return df


def inspect_raw_data(df: pd.DataFrame) -> dict:
    """Inspect initial data quality metrics: nulls, duplicates, cancellations."""
    print("[2/7] Inspecting data quality...")
    null_counts = df.isnull().sum().to_dict()
    total_duplicates = int(df.duplicated().sum())
    cancellation_count = int(df["InvoiceNo"].astype(str).str.startswith("C").sum())
    
    print(f"      Missing CustomerIDs: {null_counts.get('CustomerID', 0):,} ({null_counts.get('CustomerID', 0)/len(df):.2%})")
    print(f"      Missing Descriptions: {null_counts.get('Description', 0):,}")
    print(f"      Duplicate Rows: {total_duplicates:,}")
    print(f"      Cancellations (Invoice starting with 'C'): {cancellation_count:,}")
    
    return {
        "initial_row_count": len(df),
        "null_counts": null_counts,
        "duplicate_rows": total_duplicates,
        "cancellation_count": cancellation_count
    }


def clean_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply documented cleaning rules:
    1. Parse InvoiceDate into standard datetime.
    2. Remove duplicate transactions.
    3. Exclude cancelled transactions (InvoiceNo starting with 'C').
    4. Exclude negative and zero quantities or unit prices.
    5. Clean stock codes and whitespace in descriptions.
    6. Derive TotalPrice = Quantity * UnitPrice.
    """
    print("[3/7] Cleaning dataset based on documented business rules...")
    clean_df = df.copy()

    # 1. Parse dates
    clean_df["InvoiceDate"] = pd.to_datetime(clean_df["InvoiceDate"], errors="coerce")
    clean_df = clean_df.dropna(subset=["InvoiceDate"])

    # 2. Drop duplicates
    clean_df = clean_df.drop_duplicates()

    # 3. Filter cancellations
    clean_df["InvoiceNo"] = clean_df["InvoiceNo"].astype(str).str.strip()
    clean_df = clean_df[~clean_df["InvoiceNo"].str.startswith("C")]

    # 4. Valid positive quantities and prices
    clean_df = clean_df[(clean_df["Quantity"] > 0) & (clean_df["UnitPrice"] > 0)]

    # 5. Clean text fields
    clean_df["Description"] = clean_df["Description"].astype(str).str.strip()
    clean_df["Country"] = clean_df["Country"].astype(str).str.strip()

    # 6. Derive financial metrics
    clean_df["TotalPrice"] = clean_df["Quantity"] * clean_df["UnitPrice"]
    clean_df["YearMonth"] = clean_df["InvoiceDate"].dt.to_period("M").astype(str)
    clean_df["Hour"] = clean_df["InvoiceDate"].dt.hour
    clean_df["DayOfWeek"] = clean_df["InvoiceDate"].dt.day_name()

    print(f"      Cleaned dataset retained: {len(clean_df):,} rows ({len(clean_df)/len(df):.2%} of raw data).")
    return clean_df


def analyze_sales_and_products(df: pd.DataFrame) -> dict:
    """Analyze monthly sales trends and identify top-performing products."""
    print("[4/7] Performing revenue and product performance analysis...")
    
    # Monthly sales trend
    monthly_sales = (
        df.groupby("YearMonth")
        .agg(
            TotalRevenue=("TotalPrice", "sum"),
            TotalQuantity=("Quantity", "sum"),
            OrderCount=("InvoiceNo", "nunique")
        )
        .reset_index()
    )

    # Top 10 products by revenue
    top_products_rev = (
        df.groupby(["StockCode", "Description"])
        .agg(TotalRevenue=("TotalPrice", "sum"), TotalQuantity=("Quantity", "sum"))
        .reset_index()
        .sort_values(by="TotalRevenue", ascending=False)
        .head(10)
    )

    # Top countries by revenue
    country_sales = (
        df.groupby("Country")
        .agg(TotalRevenue=("TotalPrice", "sum"), OrderCount=("InvoiceNo", "nunique"))
        .reset_index()
        .sort_values(by="TotalRevenue", ascending=False)
    )

    return {
        "monthly_sales": monthly_sales,
        "top_products_revenue": top_products_rev,
        "country_sales": country_sales
    }


def perform_rfm_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform Recency, Frequency, Monetary (RFM) customer segmentation:
    - Recency: Days since customer's last purchase relative to study anchor date.
    - Frequency: Unique orders placed by the customer.
    - Monetary: Total spending by the customer.
    - Quartile scoring (1-4) and segment labeling.
    """
    print("[5/7] Executing customer RFM segmentation...")
    # Customer-level segmentation requires known CustomerID
    cust_df = df.dropna(subset=["CustomerID"]).copy()
    cust_df["CustomerID"] = cust_df["CustomerID"].astype(int).astype(str)

    reference_date = cust_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = cust_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda d: (reference_date - d.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum")
    ).reset_index()

    # Filter out anomalous non-positive monetary totals if any
    rfm = rfm[rfm["Monetary"] > 0]

    # Assign quartile scores (1 = lowest, 4 = highest)
    # For Recency, lower days = better score (hence reversed labels 4 to 1)
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)

    rfm["RFM_Score"] = rfm["R_Score"] * 100 + rfm["F_Score"] * 10 + rfm["M_Score"]

    # Assign actionable customer segments
    def assign_segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 3 and f >= 3 and m >= 3:
            return "Champions"
        elif r >= 3 and f >= 2:
            return "Loyal Customers"
        elif r >= 3 and f == 1:
            return "Promising / New"
        elif r == 2 and f >= 2:
            return "Needs Attention"
        elif r <= 2 and f >= 3:
            return "At Risk"
        else:
            return "Lost / Hibernating"

    rfm["Customer_Segment"] = rfm.apply(assign_segment, axis=1)
    
    segment_summary = rfm.groupby("Customer_Segment").agg(
        CustomerCount=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        TotalRevenue=("Monetary", "sum")
    ).reset_index()
    
    print(f"      Total unique customers segmented: {len(rfm):,}")
    print("      Segment distribution:")
    for _, row in segment_summary.iterrows():
        pct = (row['CustomerCount'] / len(rfm)) * 100
        print(f"        * {row['Customer_Segment']}: {int(row['CustomerCount']):,} customers ({pct:.1f}%), ${row['TotalRevenue']:,.2f}")

    return rfm


def save_visualizations(df: pd.DataFrame, sales_data: dict, rfm: pd.DataFrame, output_dir: str):
    """Generate and export publication-ready charts."""
    print(f"[6/7] Generating analytical visualizations in: {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Monthly Revenue Trend
    plt.figure(figsize=(10, 4.5))
    monthly = sales_data["monthly_sales"]
    plt.plot(monthly["YearMonth"], monthly["TotalRevenue"] / 1000, marker="o", color="#1f77b4", linewidth=2.5)
    plt.title("Monthly Revenue Trend (in Thousands GBP)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Month (Year-Month)")
    plt.ylabel("Revenue (k GBP)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_monthly_revenue_trend.png"), dpi=300)
    plt.close()

    # 2. Top 10 Products by Revenue
    plt.figure(figsize=(10, 5))
    top_p = sales_data["top_products_revenue"]
    # Truncate long descriptions
    short_labels = [desc[:28] + "..." if len(desc) > 28 else desc for desc in top_p["Description"]]
    sns.barplot(x=top_p["TotalRevenue"] / 1000, y=short_labels, palette="Blues_r", hue=short_labels, legend=False)
    plt.title("Top 10 Products by Total Revenue", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Total Revenue (k GBP)")
    plt.ylabel("Product Description")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_top_products_revenue.png"), dpi=300)
    plt.close()

    # 3. Customer Segments Distribution
    plt.figure(figsize=(8, 4.5))
    seg_counts = rfm["Customer_Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    sns.barplot(data=seg_counts, x="Count", y="Segment", palette="viridis", hue="Segment", legend=False)
    plt.title("Customer Count by RFM Segment", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Number of Customers")
    plt.ylabel("Segment")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_rfm_customer_segments.png"), dpi=300)
    plt.close()

    # 4. Hourly Order Frequency
    plt.figure(figsize=(9, 4))
    hourly = df.groupby("Hour")["InvoiceNo"].nunique().reset_index()
    sns.barplot(data=hourly, x="Hour", y="InvoiceNo", color="#4c72b0")
    plt.title("Unique Transactions by Hour of the Day", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Hour of Day (24-Hour UTC)")
    plt.ylabel("Order Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_hourly_order_volume.png"), dpi=300)
    plt.close()

    print("      Visualizations successfully exported.")


def export_findings(clean_df: pd.DataFrame, sales_data: dict, rfm: pd.DataFrame, output_dir: str):
    """Export summary metrics to JSON for documentation and validation."""
    print(f"[7/7] Exporting summary report...")
    summary_path = os.path.join(output_dir, "ecommerce_summary_metrics.json")
    
    total_revenue = float(clean_df["TotalPrice"].sum())
    total_orders = int(clean_df["InvoiceNo"].nunique())
    total_items_sold = int(clean_df["Quantity"].sum())
    avg_order_value = total_revenue / total_orders
    total_unique_customers = int(rfm["CustomerID"].nunique())

    metrics = {
        "dataset_name": "UCI Online Retail Dataset",
        "total_revenue_gbp": round(total_revenue, 2),
        "total_orders": total_orders,
        "total_items_sold": total_items_sold,
        "average_order_value_gbp": round(avg_order_value, 2),
        "total_unique_customers": total_unique_customers,
        "rfm_segments": rfm["Customer_Segment"].value_counts().to_dict(),
        "top_product": {
            "description": sales_data["top_products_revenue"].iloc[0]["Description"],
            "revenue": round(float(sales_data["top_products_revenue"].iloc[0]["TotalRevenue"]), 2)
        },
        "top_market": {
            "country": sales_data["country_sales"].iloc[0]["Country"],
            "revenue": round(float(sales_data["country_sales"].iloc[0]["TotalRevenue"]), 2),
            "share_percent": round((sales_data["country_sales"].iloc[0]["TotalRevenue"] / total_revenue) * 100, 2)
        }
    }

    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"      Summary exported to: {summary_path}")
    print("\n" + "="*50)
    print("KEY BUSINESS FINDINGS")
    print("="*50)
    print(f"Total Revenue Generated:   GBP {total_revenue:,.2f}")
    print(f"Total Unique Orders:       {total_orders:,}")
    print(f"Average Order Value (AOV): GBP {avg_order_value:,.2f}")
    print(f"Unique Customers (RFM):    {total_unique_customers:,}")
    print(f"Primary Market:            {metrics['top_market']['country']} ({metrics['top_market']['share_percent']}% of revenue)")
    print(f"Top Revenue Product:       {metrics['top_product']['description']} (GBP {metrics['top_product']['revenue']:,.2f})")
    print("="*50)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "online_retail.csv.gz")
    viz_dir = os.path.join(base_dir, "visualizations")

    raw_df = load_dataset(data_path)
    inspect_raw_data(raw_df)
    clean_df = clean_retail_data(raw_df)
    sales_data = analyze_sales_and_products(clean_df)
    rfm_df = perform_rfm_segmentation(clean_df)
    save_visualizations(clean_df, sales_data, rfm_df, viz_dir)
    export_findings(clean_df, sales_data, rfm_df, base_dir)


if __name__ == "__main__":
    main()
