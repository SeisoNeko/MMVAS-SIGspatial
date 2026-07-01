import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def analyze_predictions(csv_path="./result/test_predictions_spatial.csv", output_dir="./result/analysis"):
    print(f"Loading predictions from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Please run your test script first.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Basic Statistical Summary
    print("\n" + "="*50)
    print("BASIC STATISTICS (Real vs Predicted)")
    print("="*50)
    
    real_mean = df['real_value'].mean()
    pred_mean = df['predicted_value'].mean()
    real_std = df['real_value'].std()
    pred_std = df['predicted_value'].std()
    
    print(f"Real Values      -> Mean: {real_mean:.2f} | Std Dev: {real_std:.2f} | Min: {df['real_value'].min():.2f} | Max: {df['real_value'].max():.2f}")
    print(f"Predicted Values -> Mean: {pred_mean:.2f} | Std Dev: {pred_std:.2f} | Min: {df['predicted_value'].min():.2f} | Max: {df['predicted_value'].max():.2f}")
    
    # R-Squared Calculation
    correlation_matrix = np.corrcoef(df['real_value'], df['predicted_value'])
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    print(f"\nR-Squared (Explained Variance): {r_squared:.4f}")
    
    # 2. Error Bins (How many are within strict tolerances?)
    print("\n" + "="*50)
    print("ERROR TOLERANCE ANALYSIS")
    print("="*50)
    
    df['perc_error'] = (df['abs_error'] / df['real_value']) * 100
    
    within_1_perc = (df['perc_error'] <= 1.0).sum()
    within_5_perc = (df['perc_error'] <= 5.0).sum()
    within_10_perc = (df['perc_error'] <= 10.0).sum()
    total = len(df)
    
    print(f"Total Test Grids: {total}")
    print(f"Predictions within 1% error:  {within_1_perc} ({within_1_perc/total*100:.2f}%)")
    print(f"Predictions within 5% error:  {within_5_perc} ({within_5_perc/total*100:.2f}%)")
    print(f"Predictions within 10% error: {within_10_perc} ({within_10_perc/total*100:.2f}%)")

    # Set up Seaborn style for professional plots
    sns.set_theme(style="whitegrid")
    
    # ---------------------------------------------------------
    # PLOT 1: Real vs. Predicted Density (The "Conservative" Check)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['real_value'], label='Real Value', fill=True, color='blue', alpha=0.5)
    sns.kdeplot(df['predicted_value'], label='Predicted Value', fill=True, color='orange', alpha=0.5)
    plt.title("Distribution Shift: Real vs Predicted Land Values")
    plt.xlabel("Land Value Rate")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_value_distribution.png"), dpi=200)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: Scatter Plot (Accuracy Trend)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='real_value', y='predicted_value', data=df, alpha=0.3, edgecolor=None, color='purple')
    
    # Plot the ideal Y = X line
    min_val = min(df['real_value'].min(), df['predicted_value'].min())
    max_val = max(df['real_value'].max(), df['predicted_value'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Prediction (Y=X)')
    
    plt.title("Prediction Accuracy Scatter Plot")
    plt.xlabel("Real Land Value Rate")
    plt.ylabel("Predicted Land Value Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_scatter_accuracy.png"), dpi=200)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 3: Residual Error Distribution
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.histplot(df['error'], bins=100, kde=True, color='teal')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.title("Residual Error Distribution (Predicted - Real)")
    plt.xlabel("Error (Negative = Under-predicted, Positive = Over-predicted)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_error_distribution.png"), dpi=200)
    plt.close()

    print(f"\nAnalysis complete! Charts saved to: {output_dir}")

if __name__ == "__main__":
    analyze_predictions()