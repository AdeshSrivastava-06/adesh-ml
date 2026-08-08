"""
Robust Scaling

RobustScaler uses the median and IQR, making it useful
for numerical data containing outliers.
"""

import pandas as pd
from sklearn.preprocessing import RobustScaler

# Load dataset
df = pd.read_csv("weight-height.csv")

# Select numerical features
features = ["Height", "Weight"]

# Keep original values
original = df[features].copy()

# Apply Robust Scaling
scaler = RobustScaler()
scaled = scaler.fit_transform(df[features])

# Create comparison
result = pd.DataFrame(
    scaled,
    columns=[col + "_scaled" for col in features]
)

comparison = pd.concat(
    [original, result],
    axis=1
)

print("BEFORE vs AFTER ROBUST SCALING")
print("=" * 50)
print(comparison.head(10))

# Show statistics before scaling
print("\nStatistics BEFORE scaling:")
print(original.describe().loc[["mean", "50%"]])

# Show statistics after scaling
print("\nStatistics AFTER scaling:")
print(result.describe().loc[["mean", "50%"]])

print("\nMedian after scaling:")
print(result.median())
