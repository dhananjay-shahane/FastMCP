#!/usr/bin/env python3
"""
Line Graph Generation Script
Creates line graphs from CSV data with time series support
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

def create_line_graph(csv_file, x_column, y_column, title=None, output_dir="output", group_column=None):
    """
    Create a line graph from CSV data
    
    Args:
        csv_file (str): Path to CSV file
        x_column (str): Column name for x-axis
        y_column (str): Column name for y-axis
        title (str): Chart title
        output_dir (str): Output directory for chart
        group_column (str): Optional column to group lines by
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
        if group_column and group_column not in df.columns:
            raise ValueError(f"Group column '{group_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Convert date column if it looks like dates
        if 'date' in x_column.lower() or any(char in str(df[x_column].iloc[0]) for char in ['-', '/']):
            try:
                df[x_column] = pd.to_datetime(df[x_column])
                df = df.sort_values(x_column)
            except:
                pass  # If conversion fails, use as is
        
        # Create figure and axis
        plt.figure(figsize=(14, 8))
        
        # Create line graph
        if group_column:
            # Multiple lines grouped by column
            groups = df[group_column].unique()
            colors = plt.cm.get_cmap('Set1')(range(len(groups)))
            
            for i, group in enumerate(groups):
                group_data = df[df[group_column] == group]
                group_data = group_data.sort_values(x_column)
                
                plt.plot(group_data[x_column], group_data[y_column], 
                        marker='o', linewidth=2, markersize=6, 
                        label=str(group), color=colors[i], alpha=0.8)
            
            plt.legend(title=group_column.replace('_', ' ').title(), 
                      bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Single line
            df_sorted = df.sort_values(x_column)
            plt.plot(df_sorted[x_column], df_sorted[y_column], 
                    marker='o', linewidth=2, markersize=6, 
                    color='steelblue', alpha=0.8)
        
        # Customize chart
        if title:
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
        else:
            if group_column:
                plt.title(f'{y_column} by {x_column} (grouped by {group_column})', 
                         fontsize=16, fontweight='bold', pad=20)
            else:
                plt.title(f'{y_column} over {x_column}', fontsize=16, fontweight='bold', pad=20)
        
        plt.xlabel(x_column.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        plt.ylabel(y_column.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        
        # Format x-axis if datetime
        if pd.api.types.is_datetime64_any_dtype(df[x_column]):
            plt.xticks(rotation=45)
            import matplotlib.dates as mdates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        elif len(str(df[x_column].iloc[0])) > 8:
            plt.xticks(rotation=45, ha='right')
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        group_suffix = f"_{group_column}" if group_column else ""
        filename = f"line_graph_{x_column}_{y_column}{group_suffix}_{timestamp}.png"
        output_path = os.path.join(output_dir, filename)
        
        # Save chart
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # Print results
        print(f"SUCCESS: Line graph created successfully")
        print(f"File: {output_path}")
        print(f"Data points: {len(df)}")
        print(f"X-axis: {x_column}")
        print(f"Y-axis: {y_column}")
        if group_column:
            print(f"Groups: {df[group_column].nunique()} unique values in {group_column}")
        
        return output_path
        
    except Exception as e:
        print(f"ERROR: Failed to create line graph: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='Generate line graphs from CSV data')
    parser.add_argument('csv_file', help='CSV file path')
    parser.add_argument('x_column', help='Column name for x-axis')
    parser.add_argument('y_column', help='Column name for y-axis')
    parser.add_argument('--title', help='Chart title', default=None)
    parser.add_argument('--output-dir', help='Output directory', default='output')
    parser.add_argument('--group-by', help='Column to group lines by', default=None)
    
    # Handle both command line and script arguments
    if len(sys.argv) < 3:
        # Default example for trends data
        csv_file = "trends.csv"
        x_column = "date"
        y_column = "value"
        title = "Metrics Over Time"
        output_dir = "output"
        group_column = "metric"
    else:
        args = parser.parse_args()
        csv_file = args.csv_file
        x_column = args.x_column
        y_column = args.y_column
        title = args.title
        output_dir = args.output_dir
        group_column = args.group_by
    
    create_line_graph(csv_file, x_column, y_column, title, output_dir, group_column)

if __name__ == "__main__":
    main()
