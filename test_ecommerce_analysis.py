"""
Unit tests for E-Commerce Customer Analysis & RFM Segmentation pipeline.
"""

import os
import unittest
import pandas as pd
import numpy as np

from ecommerce_analysis import clean_retail_data, perform_rfm_segmentation, analyze_sales_and_products


class TestEcommerceAnalysis(unittest.TestCase):
    def setUp(self):
        """Create a synthetic transactional dataset covering edge cases."""
        self.sample_raw_data = pd.DataFrame({
            "InvoiceNo": ["536365", "536365", "C536366", "536367", "536368", "536368", "536369"],
            "StockCode": ["85123A", "71053", "22633", "84879", "22960", "22960", "21756"],
            "Description": [" WHITE HANGING HEART ", "WHITE METAL LANTERN", "CANCELLED ITEM", "ASSORTED COLOUR BIRD", "JAM JAR", "JAM JAR", "BATH MAT"],
            "Quantity": [6, 2, -1, 32, 12, 12, -5],
            "InvoiceDate": [
                "12/1/2010 8:26",
                "12/1/2010 8:26",
                "12/1/2010 8:28",
                "12/1/2010 8:34",
                "12/1/2010 8:34",
                "12/1/2010 8:34",  # Duplicate row
                "12/1/2010 8:35"   # Negative quantity
            ],
            "UnitPrice": [2.55, 3.39, 1.85, 1.69, 4.25, 4.25, 5.99],
            "CustomerID": [17850.0, 17850.0, 17850.0, 13047.0, 12583.0, 12583.0, 12583.0],
            "Country": ["United Kingdom", "United Kingdom", "United Kingdom", "United Kingdom", "France", "France", "France"]
        })

    def test_clean_retail_data_filtering(self):
        """Verify duplicates, cancellations, and negative quantities are removed."""
        cleaned = clean_retail_data(self.sample_raw_data)
        
        # 7 rows raw -> 1 duplicate dropped (6 left) -> 1 cancellation dropped (5 left) -> 1 negative quantity dropped (4 left)
        self.assertEqual(len(cleaned), 4)
        
        # Verify no invoice starts with 'C'
        self.assertFalse(cleaned["InvoiceNo"].str.startswith("C").any())
        
        # Verify all quantities and unit prices are strictly positive
        self.assertTrue((cleaned["Quantity"] > 0).all())
        self.assertTrue((cleaned["UnitPrice"] > 0).all())
        
        # Verify TotalPrice calculation
        expected_total = (cleaned["Quantity"] * cleaned["UnitPrice"]).sum()
        self.assertAlmostEqual(cleaned["TotalPrice"].sum(), expected_total, places=2)

    def test_rfm_segmentation_logic(self):
        """Verify RFM scoring and segment assignment on multi-customer transactions."""
        # Create multi-transaction scenario
        rfm_test_data = pd.DataFrame({
            "InvoiceNo": ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008"],
            "CustomerID": [1, 1, 2, 3, 4, 4, 4, 4],
            "InvoiceDate": pd.to_datetime([
                "2011-12-01 10:00", "2011-12-05 10:00",
                "2011-01-10 10:00",
                "2011-11-20 10:00",
                "2011-12-08 10:00", "2011-12-08 11:00", "2011-12-08 12:00", "2011-12-08 13:00"
            ]),
            "TotalPrice": [100.0, 150.0, 50.0, 75.0, 500.0, 600.0, 700.0, 800.0]
        })
        
        rfm = perform_rfm_segmentation(rfm_test_data)
        
        # Check that 4 customers were segmented
        self.assertEqual(len(rfm), 4)
        
        # Check scores are integers between 1 and 4
        for col in ["R_Score", "F_Score", "M_Score"]:
            self.assertTrue(rfm[col].between(1, 4).all())
            
        # Check that valid segments are assigned
        valid_segments = {
            "Champions", "Loyal Customers", "Promising / New",
            "Needs Attention", "At Risk", "Lost / Hibernating"
        }
        self.assertTrue(set(rfm["Customer_Segment"].unique()).issubset(valid_segments))

    def test_metrics_json_integrity(self):
        """Verify ecommerce_summary_metrics.json exists and contains correct values."""
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecommerce_summary_metrics.json")
        self.assertTrue(os.path.exists(json_path))
        
        import json
        with open(json_path, "r") as f:
            metrics = json.load(f)
            
        self.assertEqual(metrics["total_unique_customers"], 4338)
        self.assertEqual(metrics["total_orders"], 19960)
        self.assertAlmostEqual(metrics["total_revenue_gbp"], 10642110.80, places=1)
        self.assertEqual(metrics["top_market"]["country"], "United Kingdom")


if __name__ == "__main__":
    unittest.main()
