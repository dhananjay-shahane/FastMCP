#!/usr/bin/env python3
"""
Sample data analysis script
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def analyze_data(csv_file="data/sample_data.csv"):
    """Analyze sample data and create basic statistics"""
    try:
        # Read CSV file
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            print(f"Error: CSV file {csv_file} not found")
            return
        
        print("=== Data Analysis Report ===")
        print(f"Total records: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}")
        print("\n=== Basic Statistics ===")
        print(df.describe())
        
        # Save results
        os.makedirs("output", exist_ok=True)
        
        # Create a simple bar chart if salary column exists
        if 'salary' in df.columns:
            plt.figure(figsize=(10, 6))
            df.plot(x='name', y='salary', kind='bar')
            plt.title('Salary by Person')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('output/salary_chart.png')
            print("Chart saved to output/salary_chart.png")
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_data(sys.argv[1])
    else:
        analyze_data()