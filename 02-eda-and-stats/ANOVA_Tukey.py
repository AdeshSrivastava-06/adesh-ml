# One-Way ANOVA and Tukey HSD Post-Hoc Test
# Dataset: Wine Dataset from sklearn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

wine = load_wine()

df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

df["target"] = wine.target

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

#  Select Variable for Analysis

# We want to check whether the mean alcohol
# concentration differs between wine classes.

data = df[["alcohol", "target"]].copy()

data["wine_class"] = data["target"].map({
    0: "Class 0",
    1: "Class 1",
    2: "Class 2"
})

print("\nMean Alcohol by Wine Class:")
print(data.groupby("wine_class")["alcohol"].agg(
    ["count", "mean", "std", "min", "max"]
))


#  Visual EDA

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=data,
    x="wine_class",
    y="alcohol"
)

plt.title("Alcohol Distribution Across Wine Classes")
plt.xlabel("Wine Class")
plt.ylabel("Alcohol")
plt.tight_layout()
plt.show()


#  Prepare Groups for ANOVA

group_0 = data[data["target"] == 0]["alcohol"]
group_1 = data[data["target"] == 1]["alcohol"]
group_2 = data[data["target"] == 2]["alcohol"]


#  One-Way ANOVA

f_stat, p_value = f_oneway(
    group_0,
    group_1,
    group_2
)

print("\nOne-Way ANOVA Results")
print("---------------------")
print("F-statistic:", round(f_stat, 4))
print("P-value:", round(p_value, 6))


alpha = 0.05

if p_value < alpha:
    print("Result: Significant difference exists.")
else:
    print("Result: No significant difference found.")


#  Tukey HSD Post-Hoc Test

tukey = pairwise_tukeyhsd(
    endog=data["alcohol"],
    groups=data["wine_class"],
    alpha=0.05
)

print("\nTukey HSD Post-Hoc Test")
print(tukey)


# Interpret Significant Pairs

print("\nPairwise Comparison:")

for result in tukey.summary().data[1:]:
    group1 = result[0]
    group2 = result[1]
    p_adj = result[3]
    reject = result[6]

    if reject:
        print(
            f"{group1} vs {group2}: "
            f"Significant difference (p={p_adj:.4f})"
        )
    else:
        print(
            f"{group1} vs {group2}: "
            f"No significant difference (p={p_adj:.4f})"
        )


#  Effect Size: Eta Squared

grand_mean = data["alcohol"].mean()

between_group_ss = sum(
    len(group) * (group.mean() - grand_mean) ** 2
    for group in [group_0, group_1, group_2]
)

total_ss = sum(
    (data["alcohol"] - grand_mean) ** 2
)

eta_squared = between_group_ss / total_ss

print("\nEffect Size")
print("-----------")
print("Eta Squared:", round(eta_squared, 4))

if eta_squared < 0.01:
    print("Effect: Very small")
elif eta_squared < 0.06:
    print("Effect: Small")
elif eta_squared < 0.14:
    print("Effect: Medium")
else:
    print("Effect: Large")
