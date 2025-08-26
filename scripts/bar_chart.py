#!/usr/bin/env python3
"""
Bar Chart Generation Script
Creates bar charts from CSV data with customizable parameters
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

def create_bar_chart(csv_file, x_column, y_column, title=None, output_dir="output"):
    """
    Create a bar chart from CSV data
    
    Args:
        csv_file (str): Path to CSV file
        x_column (str): Column name for x-axis
        y_column (str): Column name for y-axis
        title (str): Chart title
        output_dir (str): Output directory for chart
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Read CSV file
        if not os.path.exists(csv_file):
            csv_path = os.path.join("data", csv_file)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_file}")
            csv_file = csv_path
        
        df = pd.read_csv(csv_file)
        
        # Validate columns exist
        if x_column not in df.columns:
            raise ValueError(f"Column '{x_column}' not found in CSV. Available columns: {list(df.columns)}")
        if y_column not in df.columns:
            raise ValueError(f"Column '{y_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Group data if needed (sum duplicate x values)
        if df[x_column].dtype == 'object':
            grouped_df = df.groupby(x_column)[y_column].sum().reset_index()
        else:
            grouped_df = df.copy()
        
        # Create figure and axis
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        bars = plt.bar(grouped_df[x_column], grouped_df[y_column], 
                      color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize chart
        if title:
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
        else:
            plt.title(f'{y_column} by {x_column}', fontsize=16, fontweight='bold', pad=20)
        
        plt.xlabel(x_column.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        plt.ylabel(y_column.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:,.0f}', ha='center', va='bottom', fontsize=10)
        
        # Rotate x-axis labels if needed
        if len(str(grouped_df[x_column].iloc[0])) > 8:
            plt.xticks(rotation=45, ha='right')
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bar_chart_{x_column}_{y_column}_{timestamp}.png"
        output_path = os.path.join(output_dir, filename)
        
        # Save chart
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # Print results
        print(f"SUCCESS: Bar chart created successfully")
        print(f"File: {output_path}")
        print(f"Data points: {len(grouped_df)}")
        print(f"X-axis: {x_column}")
        print(f"Y-axis: {y_column}")
        
        return output_path
        
    except Exception as e:
        print(f"ERROR: Failed to create bar chart: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='Generate bar charts from CSV data')
    parser.add_argument('csv_file', help='CSV file path')
    parser.add_argument('x_column', help='Column name for x-axis')
    parser.add_argument('y_column', help='Column name for y-axis')
    parser.add_argument('--title', help='Chart title', default=None)
    parser.add_argument('--output-dir', help='Output directory', default='output')
    
    # Handle both command line and script arguments
    if len(sys.argv) < 3:
        # Default example for sales data
        csv_file = "sales.csv"
        x_column = "category"
        y_column = "sales_amount"
        title = "Sales by Category"
        output_dir = "output"
    else:
        args = parser.parse_args()
        csv_file = args.csv_file
        x_column = args.x_column
        y_column = args.y_column
        title = args.title
        output_dir = args.output_dir
    
    create_bar_chart(csv_file, x_column, y_column, title, output_dir)

if __name__ == "__main__":
    main()
