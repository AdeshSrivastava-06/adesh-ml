import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# IQR method is used to detect and handle outliers
# Unlike Z-score, IQR does not assume that data is normally distributed
# This makes it more reliable for skewed data

# IQR = Q3 - Q1 (75th percentile - 25th percentile)
# Any value below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is considered an outlier

# Load dataset
df = pd.read_csv('placement.csv')

# Select the numerical column to check for outliers
column = 'cgpa'

# Boxplot before treatment to visually see outliers
plt.figure(figsize=(6, 4))
plt.boxplot(df[column])
plt.title('Before Outlier Treatment')
plt.savefig('before_iqr_treatment.png')
plt.show()

# Calculate Q1 and Q3
Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Calculate lower and upper boundary
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

# Find outliers before treatment
outliers = df[(df[column] < lower_limit) | (df[column] > upper_limit)]
print("Number of outliers found:", outliers.shape[0])

# Method 1: Trimming (remove the outlier rows completely)
df_trimmed = df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]
print("Shape before trimming:", df.shape)
print("Shape after trimming:", df_trimmed.shape)

# Method 2: Capping (replace outliers with the boundary values instead of removing)
df_capped = df.copy()
df_capped[column] = np.where(
    df_capped[column] > upper_limit,
    upper_limit,
    np.where(
        df_capped[column] < lower_limit,
        lower_limit,
        df_capped[column]
    )
)

# Boxplot after capping to compare
plt.figure(figsize=(6, 4))
plt.boxplot(df_capped[column])
plt.title('After Outlier Capping')
plt.savefig('after_iqr_treatment.png')
plt.show()

# Use df_trimmed if you can afford to lose rows
# Use df_capped if you want to keep all rows but limit extreme values
