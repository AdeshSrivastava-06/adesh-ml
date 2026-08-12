import pandas as pd
from sklearn.preprocessing import TargetEncoder

# Sample dataset
df = pd.DataFrame({
    "City": ["Mumbai", "Delhi", "Mumbai", "Pune", "Delhi",
             "Pune", "Mumbai", "Delhi"],
    "Age": [21, 25, 22, 30, 28, 35, 24, 26],
    "Purchased": [1, 1, 0, 0, 1, 1, 1, 0]
})

print("BEFORE TARGET ENCODING")
print(df)

# Separate features and target
X = df[["City", "Age"]]
y = df["Purchased"]

# Create target encoder
encoder = TargetEncoder(
    target_type="binary",
    random_state=42
)

# Encode City using the target
X_encoded = X.copy()
X_encoded["City"] = encoder.fit_transform(X[["City"]], y)

print("\nAFTER TARGET ENCODING")
print(X_encoded)

# Show the learned mapping
mapping = pd.DataFrame({
    "City": encoder.categories_[0],
    "Target_Mean": encoder.encodings_[0]
})

print("\nTARGET ENCODING MAPPING")
print(mapping)
