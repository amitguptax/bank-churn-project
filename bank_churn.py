import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate sample data
np.random.seed(42)

n = 500

df = pd.DataFrame({
    "age": np.random.randint(18, 70, n),
    "balance": np.random.randint(1000, 200000, n),
    "products": np.random.randint(1, 5, n),
    "active": np.random.randint(0, 2, n),
    "salary": np.random.randint(15000, 150000, n)
})

# 2. Generate sample churn labels
probability = (
    0.10
    + 0.25 * (df["active"] == 0)
    + 0.15 * (df["products"] == 1)
    + 0.10 * (df["balance"] < 20000)
)

probability = np.clip(probability, 0, 0.95)
df["churn"] = np.random.binomial(1, probability)

print("\n===== BANK CHURN PROJECT =====")
print("\nFirst five customers:")
print(df.head())

# 3. Data analysis using Pandas
print("\n===== DATA ANALYSIS =====")
print("Total customers:", len(df))
print("Churned customers:", int(df["churn"].sum()))
print("Retained customers:", int((df["churn"] == 0).sum()))
print("Average balance:", round(df["balance"].mean(), 2))
print("Churn percentage:", round(df["churn"].mean() * 100, 2), "%")

# 4. Filter inactive customers
print("\n===== INACTIVE CUSTOMERS =====")
print(df[df["active"] == 0].head(10))

# 5. NumPy calculations
print("\n===== NUMPY ANALYSIS =====")
balances = df["balance"].to_numpy()
print("Minimum balance:", np.min(balances))
print("Maximum balance:", np.max(balances))
print("Median balance:", np.median(balances))

# 6. Prepare machine learning data
features = ["age", "balance", "products", "active", "salary"]
X = df[features]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 7. Train machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 8. Evaluate model
predictions = model.predict(X_test)

print("\n===== MODEL EVALUATION =====")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print(classification_report(y_test, predictions, zero_division=0))

# 9. Predict for a new sample customer
new_customer = pd.DataFrame([{
    "age": 30,
    "balance": 15000,
    "products": 1,
    "active": 0,
    "salary": 30000
}])

prediction = model.predict(new_customer)[0]
churn_probability = model.predict_proba(new_customer)[0][1]

print("\n===== CUSTOMER PREDICTION =====")
print("Predicted churn:", "Yes" if prediction == 1 else "No")
print("Estimated churn probability:", round(churn_probability * 100, 2), "%")

# 10. Feature importance
importance = pd.Series(
    model.feature_importances_, index=features
).sort_values(ascending=True)

print("\n===== FEATURE IMPORTANCE =====")
print(importance)

# 11. Create charts
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df["churn"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0]
)
axes[0].set_title("Customer Churn Distribution")
axes[0].set_xlabel("Churn: 0 = Retained, 1 = Churned")
axes[0].set_ylabel("Number of Customers")

importance.plot(kind="barh", ax=axes[1])
axes[1].set_title("Feature Importance")

plt.tight_layout()
plt.savefig("churn_analysis.png", dpi=150)
plt.show()

# 12. Save dataset
df.to_csv("bank_customers.csv", index=False)

print("\nDataset saved as bank_customers.csv")
print("Chart saved as churn_analysis.png")
print("\nProject completed successfully!")