# AdaBoost Classifier
# AdaBoost stands for Adaptive Boosting
# It combines many weak learners, usually shallow decision trees
# Each new learner focuses more on the samples the previous learners got wrong
# The final prediction is a weighted vote of all the weak learners

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
# Using heart.csv since it is already present in this repo
df = pd.read_csv("heart.csv")

# Split features and target
# Assuming the last column is the target, change this if your target column is named differently
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the weak learner
# A decision stump is a decision tree with depth 1
# AdaBoost usually works well with very simple weak learners
weak_learner = DecisionTreeClassifier(max_depth=1)

# Create the AdaBoost model
# n_estimators is the number of weak learners to combine
# learning_rate controls how much each learner contributes to the final result
ada_model = AdaBoostClassifier(
    estimator=weak_learner,
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

# Train the model
ada_model.fit(X_train, y_train)

# Predict on the test set
y_pred = ada_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Check how important each feature is
# AdaBoost gives a feature importance score based on how much each feature helped reduce error
feature_importance = pd.Series(
    ada_model.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Try different numbers of estimators to see how accuracy changes
# This helps understand if the model is underfitting or overfitting as we add more learners
print("\nAccuracy vs number of estimators:")
for n in [10, 25, 50, 100, 150, 200]:
    model = AdaBoostClassifier(
        estimator=weak_learner,
        n_estimators=n,
        learning_rate=1.0,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("n_estimators:", n, "accuracy:", acc)
