from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
dataset = load_dataset("aaronjpi/gold-price-5-years")
dataframe = pd.DataFrame(dataset['train'])
df = dataframe[['Open','Date']]

# Convert Open to float (it's stored as string)
df['Open'] = pd.to_numeric(df['Open'], errors='coerce')

# Data processing
print("Dataset Information:")
print(df.head())
print(df.describe())
print(df.dtypes)

df["Date"] = pd.to_datetime(df["Date"])
df['Year'] = df["Date"].dt.year
df['Month'] = df["Date"].dt.month
df['Day'] = df["Date"].dt.day

print("\nTransformed DataFrame:")
print(df.head())

# Feature selection
X = df[['Year','Month','Day']]
Y = df['Open']

# Train-test split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train,Y_train)
lr_predictions = lr_model.predict(X_test)

lr_mse = mean_squared_error(Y_test, lr_predictions)
lr_mae = mean_absolute_error(Y_test, lr_predictions)

print("\nLinear Regression Model")
print(f"Mean squared error: ${lr_mse:.2f}")
print(f"Mean absolute error: ${lr_mae:.2f}")

# Random Forest Regression
print("\nRandom Forest Regressor Model")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)
rf_predictions = rf_model.predict(X_test)

rf_mse = mean_squared_error(Y_test, rf_predictions)
rf_mae = mean_absolute_error(Y_test, rf_predictions)

print(f"Mean squared error: ${rf_mse:.2f}")
print(f"Mean absolute error: ${rf_mae:.2f}")

# Plot Actual vs Predicted for Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, rf_predictions, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual Prices ($)")
plt.ylabel("Predicted Prices ($)")
plt.title("Actual vs Predicted Gold Prices (Random Forest)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("actual_vs_predicted_rf.png", dpi=300)
print("\n✓ Saved: actual_vs_predicted_rf.png")
plt.close()

# Residual Plot for Random Forest
rf_residuals = Y_test - rf_predictions
plt.figure(figsize=(10, 6))
plt.scatter(rf_predictions, rf_residuals, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel("Predicted Prices ($)")
plt.ylabel("Residuals ($)")
plt.title("Residual Plot (Random Forest)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("residual_plot_rf.png", dpi=300)
print("✓ Saved: residual_plot_rf.png")
plt.close()

# Plot Actual vs Predicted for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, lr_predictions, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual Prices ($)")
plt.ylabel("Predicted Prices ($)")
plt.title("Actual vs Predicted Gold Prices (Linear Regression)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("actual_vs_predicted_lr.png", dpi=300)
print(" Saved: actual_vs_predicted_lr.png")
plt.close()

# Residual Plot for Linear Regression
lr_residuals = Y_test - lr_predictions
plt.figure(figsize=(10, 6))
plt.scatter(lr_predictions, lr_residuals, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel("Predicted Prices ($)")
plt.ylabel("Residuals ($)")
plt.title("Residual Plot (Linear Regression)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("residual_plot_lr.png", dpi=300)
print(" Saved: residual_plot_lr.png")
plt.close()

print("\n✓ All 4 plots saved! Check your folder for PNG files.")