"""
Cyclical Encoding

Some features are cyclical in nature - hour of day, day of week, month,
day of year etc. If we encode them as plain integers (0-23 for hour,
1-12 for month), the model sees 23 and 0 as far apart, when in reality
they are right next to each other (11 PM and 12 AM)

Cyclical Encoding fixes this by projecting the feature onto a circle
using sine and cosine transforms:

    sin_feature = sin(2 * pi * value / max_value)
    cos_feature = cos(2 * pi * value / max_value)

This preserves the "closeness" of values at the boundary (eg- hour 23
and hour 0 end up close together in sin-cos space)
"""

import numpy as np
import pandas as pd


# 1. Create a synthetic dataset with cyclical columns

np.random.seed(42)

n_samples = 15
df = pd.DataFrame({
    'hour': np.random.randint(0, 24, n_samples),
    'month': np.random.randint(1, 13, n_samples),
    'day_of_week': np.random.randint(0, 7, n_samples)
})

print("Original Data:\n", df)


# 2. Generic cyclical encoding function

def cyclical_encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    return data


# 3. Apply to each cyclical column with its correct period

df = cyclical_encode(df, 'hour', 24)          # hours: 0-23
df = cyclical_encode(df, 'month', 12)         # months: 1-12
df = cyclical_encode(df, 'day_of_week', 7)    # days: 0-6

print("\nAfter Cyclical Encoding:\n", df)


# 4. Sanity check - hour 23 and hour 0 should be close in sin-cos space

h23 = cyclical_encode(pd.DataFrame({'hour': [23]}), 'hour', 24)
h0 = cyclical_encode(pd.DataFrame({'hour': [0]}), 'hour', 24)

dist = np.sqrt(
    (h23['hour_sin'].values[0] - h0['hour_sin'].values[0]) ** 2 +
    (h23['hour_cos'].values[0] - h0['hour_cos'].values[0]) ** 2
)

print(f"\nDistance between hour=23 and hour=0 in cyclical space: {dist:.4f}")
