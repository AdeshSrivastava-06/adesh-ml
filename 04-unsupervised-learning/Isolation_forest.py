# Isolation Forest - Anomaly Detection

# Isolation Forest finds "weird" points (outliers/anomalies) in data.
# Idea: outliers are easy to isolate (separate) from the rest of the data
# using random splits, so they need fewer splits to get isolated.
# Normal points are packed close together, so they need more splits.
# Fewer splits needed = shorter path in the tree = more likely to be an anomaly.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# Step 1: Create sample data

# Most points are "normal" and clustered together.
# A few points are added far away to act as anomalies.

np.random.seed(42)

# 200 normal points around (0,0)
normal_data = np.random.randn(200, 2) * 1.5

# 10 anomaly points scattered far away
anomaly_data = np.random.uniform(low=-10, high=10, size=(10, 2))

# combine normal + anomaly points into one dataset
data = np.vstack([normal_data, anomaly_data])
df = pd.DataFrame(data, columns=["feature_1", "feature_2"])

print("Total points:", len(df))
print(df.head())


# Step 2: Scale the data

# Scaling helps when features have different ranges.
# Not strictly needed for Isolation Forest, but good practice.

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)


# Step 3: Train Isolation Forest

# contamination = expected proportion of anomalies in the data
# Here we guess around 5% of points are anomalies.

model = IsolationForest(
    n_estimators=100,     # number of trees in the forest
    contamination=0.05,   # expected fraction of outliers
    random_state=42
)

model.fit(scaled_data)


# Step 4: Predict anomalies

# predict() gives:
#   1  -> normal point
#  -1  -> anomaly (outlier)

df["anomaly"] = model.predict(scaled_data)

# anomaly_score: lower score = more abnormal
df["anomaly_score"] = model.decision_function(scaled_data)

print("\nNumber of anomalies detected:", (df["anomaly"] == -1).sum())
print("\nSample anomalies:")
print(df[df["anomaly"] == -1].head())


# Step 5: Visualize the results

# Normal points in blue, anomalies in red.

plt.figure(figsize=(8, 6))

normal_points = df[df["anomaly"] == 1]
anomaly_points = df[df["anomaly"] == -1]

plt.scatter(normal_points["feature_1"], normal_points["feature_2"],
            c="blue", label="Normal", alpha=0.6)

plt.scatter(anomaly_points["feature_1"], anomaly_points["feature_2"],
            c="red", label="Anomaly", marker="x", s=100)

plt.title("Isolation Forest - Anomaly Detection")
plt.xlabel("feature_1")
plt.ylabel("feature_2")
plt.legend()
plt.tight_layout()
plt.savefig("isolation_forest_plot.png")
plt.show()


# Notes

# - Isolation Forest works well for high-dimensional data too.
# - contamination is a rough guess; tune it based on your domain knowledge.
# - Unlike DBSCAN, it does not need distance-based density calculations,
#   so it scales better on large datasets.
# - Common use cases: fraud detection, network intrusion detection,
#   sensor fault detection, detecting bad data entries.
