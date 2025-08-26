#!/usr/bin/env python3
"""
Custom Statistics Generation Script
Generates comprehensive statistical analysis and reports from CSV data
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
from scipy import stats

def generate_statistics_report(csv_file, output_dir="output", columns=None):
    """
    Generate comprehensive statistics report from CSV data
    
    Args:
        csv_file (str): Path to CSV file
        output_dir (str): Output directory for reports
        columns (list): Specific columns to analyze (None for all numeric columns)
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
        
        # Filter to specified columns or numeric columns
        if columns:
            # Validate specified columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}. Available: {list(df.columns)}")
            numeric_df = df[columns].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found for statistical analysis")
        
        # Generate timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Basic Statistics Report
        stats_report = generate_basic_stats(df, numeric_df, timestamp, output_dir)
        
        # 2. Correlation Analysis
        correlation_report = generate_correlation_analysis(numeric_df, timestamp, output_dir)
        
        # 3. Distribution Analysis
        distribution_report = generate_distribution_analysis(numeric_df, timestamp, output_dir)
        
        # 4. Summary Dashboard
        dashboard_report = create_summary_dashboard(df, numeric_df, timestamp, output_dir)
        
        # Combine all reports
        full_report = {
            'basic_stats': stats_report,
            'correlation': correlation_report,
            'distribution': distribution_report,
            'dashboard': dashboard_report,
            'summary': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numeric_columns': len(numeric_df.columns),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        # Save comprehensive report as text
        report_path = os.path.join(output_dir, f"statistics_report_{timestamp}.txt")
        save_text_report(full_report, report_path)
        
        # Print summary
        print(f"SUCCESS: Statistical analysis completed")
        print(f"Report file: {report_path}")
        print(f"Dashboard: {dashboard_report}")
        print(f"Analyzed {len(numeric_df.columns)} numeric columns from {len(df)} rows")
        
        return full_report
        
    except Exception as e:
        print(f"ERROR: Failed to generate statistics: {str(e)}", file=sys.stderr)
        sys.exit(1)

def generate_basic_stats(df, numeric_df, timestamp, output_dir):
    """Generate basic descriptive statistics"""
    try:
        stats_dict = {}
        
        # Overall dataset info
        stats_dict['dataset_info'] = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Descriptive statistics for numeric columns
        desc_stats = numeric_df.describe()
        stats_dict['descriptive_stats'] = desc_stats.to_dict()
        
        # Additional statistics
        for col in numeric_df.columns:
            stats_dict[f'{col}_additional'] = {
                'variance': numeric_df[col].var(),
                'skewness': stats.skew(numeric_df[col].dropna()),
                'kurtosis': stats.kurtosis(numeric_df[col].dropna()),
                'mode': numeric_df[col].mode().iloc[0] if not numeric_df[col].mode().empty else None,
                'range': numeric_df[col].max() - numeric_df[col].min(),
                'iqr': numeric_df[col].quantile(0.75) - numeric_df[col].quantile(0.25)
            }
        
        return stats_dict
        
    except Exception as e:
        print(f"Warning: Error in basic stats generation: {str(e)}")
        return {}

def generate_correlation_analysis(numeric_df, timestamp, output_dir):
    """Generate correlation analysis and heatmap"""
    try:
        if len(numeric_df.columns) < 2:
            return {"message": "Need at least 2 numeric columns for correlation analysis"}
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
        
        plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        corr_path = os.path.join(output_dir, f"correlation_heatmap_{timestamp}.png")
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'heatmap_file': corr_path
        }
        
    except Exception as e:
        print(f"Warning: Error in correlation analysis: {str(e)}")
        return {}

def generate_distribution_analysis(numeric_df, timestamp, output_dir):
    """Generate distribution analysis with histograms"""
    try:
        distributions = {}
        
        # Create subplot grid for histograms
        n_cols = min(3, len(numeric_df.columns))
        n_rows = (len(numeric_df.columns) + n_cols - 1) // n_cols
        
        if n_rows > 0 and n_cols > 0:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_df.columns):
                if i < len(axes):
                    ax = axes[i]
                    
                    # Create histogram
                    ax.hist(numeric_df[col].dropna(), bins=30, alpha=0.7, 
                           color='steelblue', edgecolor='black', linewidth=0.5)
                    ax.set_title(f'Distribution of {col}', fontweight='bold')
                    ax.set_xlabel(col.replace('_', ' ').title())
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics text
                    mean_val = numeric_df[col].mean()
                    std_val = numeric_df[col].std()
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                    ax.legend()
                    
                    # Store distribution info
                    distributions[col] = {
                        'mean': mean_val,
                        'std': std_val,
                        'min': numeric_df[col].min(),
                        'max': numeric_df[col].max(),
                        'normality_test': stats.normaltest(numeric_df[col].dropna()).pvalue
                    }
            
            # Hide empty subplots
            for j in range(len(numeric_df.columns), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            dist_path = os.path.join(output_dir, f"distributions_{timestamp}.png")
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'distributions': distributions,
                'histogram_file': dist_path
            }
        else:
            return {"message": "No numeric columns for distribution analysis"}
        
    except Exception as e:
        print(f"Warning: Error in distribution analysis: {str(e)}")
        return {}

def create_summary_dashboard(df, numeric_df, timestamp, output_dir):
    """Create a comprehensive summary dashboard"""
    try:
        # Create a 2x2 dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Data overview (top-left)
        ax1.text(0.1, 0.9, 'Dataset Overview', fontsize=16, fontweight='bold', transform=ax1.transAxes)
        overview_text = f"""
        Total Rows: {len(df):,}
        Total Columns: {len(df.columns)}
        Numeric Columns: {len(numeric_df.columns)}
        Missing Values: {df.isnull().sum().sum()}
        Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB
        
        Column Types:
        {df.dtypes.value_counts().to_string()}
        """
        ax1.text(0.1, 0.1, overview_text, fontsize=10, transform=ax1.transAxes, 
                verticalalignment='bottom', fontfamily='monospace')
        ax1.axis('off')
        
        # 2. Missing values heatmap (top-right)
        if df.isnull().sum().sum() > 0:
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                ax2.bar(range(len(missing_data)), missing_data.values, color='lightcoral')
                ax2.set_title('Missing Values by Column', fontweight='bold')
                ax2.set_xlabel('Columns')
                ax2.set_ylabel('Missing Count')
                ax2.set_xticks(range(len(missing_data)))
                ax2.set_xticklabels(missing_data.index, rotation=45, ha='right')
            else:
                ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=14)
                ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.axis('off')
        
        # 3. Numeric columns summary (bottom-left)
        if not numeric_df.empty:
            summary_stats = numeric_df.describe().loc[['mean', 'std']].T
            x_pos = range(len(summary_stats))
            
            ax3.bar([x - 0.2 for x in x_pos], summary_stats['mean'], 0.4, 
                   label='Mean', alpha=0.8, color='steelblue')
            ax3.bar([x + 0.2 for x in x_pos], summary_stats['std'], 0.4, 
                   label='Std Dev', alpha=0.8, color='orange')
            
            ax3.set_title('Mean and Standard Deviation', fontweight='bold')
            ax3.set_xlabel('Numeric Columns')
            ax3.set_ylabel('Value')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(summary_stats.index, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Numeric Columns', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14)
            ax3.axis('off')
        
        # 4. Data quality score (bottom-right)
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        uniqueness = df.nunique().mean() / len(df) * 100 if len(df) > 0 else 0
        consistency = 100  # Simplified - could add more sophisticated checks
        
        quality_scores = [completeness, uniqueness, consistency]
        quality_labels = ['Completeness', 'Uniqueness', 'Consistency']
        colors = ['green' if score >= 80 else 'orange' if score >= 60 else 'red' for score in quality_scores]
        
        bars = ax4.bar(quality_labels, quality_scores, color=colors, alpha=0.7)
        ax4.set_title('Data Quality Metrics', fontweight='bold')
        ax4.set_ylabel('Score (%)')
        ax4.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, score in zip(bars, quality_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        dashboard_path = os.path.join(output_dir, f"summary_dashboard_{timestamp}.png")
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return dashboard_path
        
    except Exception as e:
        print(f"Warning: Error creating dashboard: {str(e)}")
        return None

def save_text_report(report_data, file_path):
    """Save comprehensive text report"""
    try:
        with open(file_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE STATISTICAL ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset Summary
            if 'summary' in report_data:
                f.write("DATASET SUMMARY\n")
                f.write("-"*40 + "\n")
                summary = report_data['summary']
                for key, value in summary.items():
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")
            
            # Basic Statistics
            if 'basic_stats' in report_data and 'descriptive_stats' in report_data['basic_stats']:
                f.write("DESCRIPTIVE STATISTICS\n")
                f.write("-"*40 + "\n")
                desc_stats = report_data['basic_stats']['descriptive_stats']
                df_stats = pd.DataFrame(desc_stats)
                f.write(df_stats.to_string())
                f.write("\n\n")
            
            # Strong Correlations
            if 'correlation' in report_data and 'strong_correlations' in report_data['correlation']:
                f.write("STRONG CORRELATIONS (|r| > 0.7)\n")
                f.write("-"*40 + "\n")
                strong_corrs = report_data['correlation']['strong_correlations']
                if strong_corrs:
                    for corr in strong_corrs:
                        f.write(f"{corr['var1']} <-> {corr['var2']}: {corr['correlation']:.3f}\n")
                else:
                    f.write("No strong correlations found.\n")
                f.write("\n")
            
            # Distribution Analysis
            if 'distribution' in report_data and 'distributions' in report_data['distribution']:
                f.write("DISTRIBUTION ANALYSIS\n")
                f.write("-"*40 + "\n")
                distributions = report_data['distribution']['distributions']
                for col, dist_info in distributions.items():
                    f.write(f"\n{col}:\n")
                    f.write(f"  Mean: {dist_info['mean']:.3f}\n")
                    f.write(f"  Std Dev: {dist_info['std']:.3f}\n")
                    f.write(f"  Range: {dist_info['min']:.3f} to {dist_info['max']:.3f}\n")
                    f.write(f"  Normality p-value: {dist_info['normality_test']:.3f}\n")
                f.write("\n")
            
        print(f"Text report saved: {file_path}")
        
    except Exception as e:
        print(f"Warning: Could not save text report: {str(e)}")

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='Generate comprehensive statistics from CSV data')
    parser.add_argument('csv_file', help='CSV file path')
    parser.add_argument('--output-dir', help='Output directory', default='output')
    parser.add_argument('--columns', nargs='+', help='Specific columns to analyze', default=None)
    
    # Handle both command line and script arguments
    if len(sys.argv) < 2:
        # Default example
        csv_file = "sales.csv"
        output_dir = "output"
        columns = None
    else:
        args = parser.parse_args()
        csv_file = args.csv_file
        output_dir = args.output_dir
        columns = args.columns
    
    generate_statistics_report(csv_file, output_dir, columns)

if __name__ == "__main__":
    main()
