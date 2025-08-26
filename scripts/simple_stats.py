#!/usr/bin/env python3
"""
Simple statistics script
"""

import pandas as pd
import sys
import os

def calculate_stats(csv_file="data/sample_data.csv"):
    """Calculate basic statistics for CSV data"""
    try:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            print(f"Error: CSV file {csv_file} not found")
            return
        
        print("=== Simple Statistics ===")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric columns: {', '.join(numeric_cols)}")
            for col in numeric_cols:
                print(f"{col}: Mean={df[col].mean():.2f}, Max={df[col].max()}, Min={df[col].min()}")
        
        print("Statistics calculation completed!")
        
    except Exception as e:
        print(f"Error calculating statistics: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        calculate_stats(sys.argv[1])
    else:
        calculate_stats()