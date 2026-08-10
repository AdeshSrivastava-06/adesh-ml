"""
Outlier Detection using IQR and Z-Score methods
-------------------------------------------------
Detects and visualizes outliers in numerical features of the Titanic dataset
(train.csv) using two standard EDA techniques:
    1. IQR (Interquartile Range) Method
    2. Z-Score Method

Both methods are compared side-by-side on the same features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data

df = pd.read_csv("train.csv")

# Columns to check for outliers
numeric_cols = ["Age", "Fare"]
df = df.dropna(subset=numeric_cols)


# 2. IQR Method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


# 3. Z-Score Method

def detect_outliers_zscore(data, column, threshold=3):
    mean = data[column].mean()
    std = data[column].std()

    z_scores = (data[column] - mean) / std
    outliers = data[np.abs(z_scores) > threshold]
    return outliers, z_scores


# 4. Run Detection + Print Summary

print("=" * 60)
print("OUTLIER DETECTION SUMMARY")
print("=" * 60)

for col in numeric_cols:
    iqr_outliers, lb, ub = detect_outliers_iqr(df, col)
    z_outliers, z_scores = detect_outliers_zscore(df, col)

    print(f"\nFeature: {col}")
    print(f"  IQR bounds        : [{lb:.2f}, {ub:.2f}]")
    print(f"  IQR outliers      : {len(iqr_outliers)} rows "
          f"({len(iqr_outliers) / len(df) * 100:.2f}%)")
    print(f"  Z-score outliers  : {len(z_outliers)} rows "
          f"({len(z_outliers) / len(df) * 100:.2f}%) [|z| > 3]")



# 5. Visualization

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for i, col in enumerate(numeric_cols):
    # Boxplot (visual IQR outliers)
    sns.boxplot(x=df[col], ax=axes[i, 0], color="skyblue")
    axes[i, 0].set_title(f"{col} - Boxplot (IQR Method)")

    # Scatter with Z-score threshold highlighted
    _, z_scores = detect_outliers_zscore(df, col)
    colors = np.where(np.abs(z_scores) > 3, "red", "steelblue")
    axes[i, 1].scatter(range(len(df)), df[col], c=colors, alpha=0.6, s=15)
    axes[i, 1].axhline(df[col].mean(), color="black", linestyle="--", linewidth=1, label="Mean")
    axes[i, 1].set_title(f"{col} - Z-Score Method (red = outlier, |z|>3)")
    axes[i, 1].legend()

plt.tight_layout()
plt.savefig("outlier_detection.png", dpi=150)
plt.show()

print("\nPlot saved as 'outlier_detection.png'")
