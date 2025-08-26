#!/usr/bin/env python3
"""
Pie Chart Generation Script
Creates pie charts from CSV data with customizable parameters
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def create_pie_chart(csv_file, category_column, value_column, title=None, output_dir="output", top_n=None):
    """
    Create a pie chart from CSV data
    
    Args:
        csv_file (str): Path to CSV file
        category_column (str): Column name for categories
        value_column (str): Column name for values
        title (str): Chart title
        output_dir (str): Output directory for chart
        top_n (int): Show only top N categories, group others as "Other"
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
        if category_column not in df.columns:
            raise ValueError(f"Column '{category_column}' not found in CSV. Available columns: {list(df.columns)}")
        if value_column not in df.columns:
            raise ValueError(f"Column '{value_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Group data by category and sum values
        grouped_df = df.groupby(category_column)[value_column].sum().reset_index()
        grouped_df = grouped_df.sort_values(value_column, ascending=False)
        
        # Handle top_n filtering
        if top_n and len(grouped_df) > top_n:
            top_categories = grouped_df.head(top_n)
            other_value = grouped_df.tail(len(grouped_df) - top_n)[value_column].sum()
            
            if other_value > 0:
                other_row = pd.DataFrame({
                    category_column: ['Other'],
                    value_column: [other_value]
                })
                grouped_df = pd.concat([top_categories, other_row], ignore_index=True)
            else:
                grouped_df = top_categories
        
        # Create figure and axis
        plt.figure(figsize=(12, 10))
        
        # Generate colors
        colors = plt.cm.get_cmap('Set3')(range(len(grouped_df)))
        
        # Create pie chart
        pie_result = plt.pie(
            grouped_df[value_column], 
            labels=grouped_df[category_column],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10},
            pctdistance=0.85
        )
        
        # Extract components
        if len(pie_result) == 3:
            wedges, texts, autotexts = pie_result
        else:
            wedges, texts = pie_result
            autotexts = []
        
        # Customize text properties
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        # Add a white circle at the center to create a donut chart effect
        from matplotlib.patches import Circle
        centre_circle = Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        # Customize chart
        if title:
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
        else:
            plt.title(f'{value_column} Distribution by {category_column}', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Add legend with values
        legend_labels = []
        for i, row in grouped_df.iterrows():
            category = row[category_column]
            value = row[value_column]
            percentage = (value / grouped_df[value_column].sum()) * 100
            legend_labels.append(f'{category}: {value:,.0f} ({percentage:.1f}%)')
        
        plt.legend(wedges, legend_labels, title="Categories", 
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=10)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        plt.axis('equal')
        
        # Tight layout to prevent cutoff
        plt.tight_layout()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pie_chart_{category_column}_{value_column}_{timestamp}.png"
        output_path = os.path.join(output_dir, filename)
        
        # Save chart
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # Print results
        print(f"SUCCESS: Pie chart created successfully")
        print(f"File: {output_path}")
        print(f"Categories: {len(grouped_df)}")
        print(f"Total value: {grouped_df[value_column].sum():,.0f}")
        print(f"Largest category: {grouped_df.iloc[0][category_column]} ({grouped_df.iloc[0][value_column]:,.0f})")
        
        return output_path
        
    except Exception as e:
        print(f"ERROR: Failed to create pie chart: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='Generate pie charts from CSV data')
    parser.add_argument('csv_file', help='CSV file path')
    parser.add_argument('category_column', help='Column name for categories')
    parser.add_argument('value_column', help='Column name for values')
    parser.add_argument('--title', help='Chart title', default=None)
    parser.add_argument('--output-dir', help='Output directory', default='output')
    parser.add_argument('--top-n', type=int, help='Show only top N categories', default=None)
    
    # Handle both command line and script arguments
    if len(sys.argv) < 3:
        # Default example for sales data
        csv_file = "sales.csv"
        category_column = "region"
        value_column = "sales_amount"
        title = "Sales Distribution by Region"
        output_dir = "output"
        top_n = None
    else:
        args = parser.parse_args()
        csv_file = args.csv_file
        category_column = args.category_column
        value_column = args.value_column
        title = args.title
        output_dir = args.output_dir
        top_n = args.top_n
    
    create_pie_chart(csv_file, category_column, value_column, title, output_dir, top_n)

if __name__ == "__main__":
    main()
