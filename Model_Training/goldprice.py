from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor


dataset = load_dataset("aaronjpi/gold-price-5-years")

dataframe = pd.DataFrame(dataset['train'])
df = dataframe[['Open','Date']]

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

X = df[['Year','Month','Day']]
Y = df['Open']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


lr_model = LinearRegression()
lr_model.fit(X_train,Y_train)

lr_predictions = lr_model.predict(X_test)

print("Linear Regression Model")

lr_mse = mean_squared_error(Y_test, lr_predictions)
lr_mae = mean_absolute_error(Y_test, lr_predictions)

print(f"Mean squared error: ${lr_mse:.2f}")
print(f"Mean absolute error: ${lr_mae:.2f}")

print("Random Forest Regressor Model")
rf_model = RandomForestRegressor()
rf_model.fit(X_train, Y_train)

rf_predictions = rf_model.predict(X_test)

rf_mse = mean_squared_error(Y_test,rf_predictions)
rf_mae = mean_absolute_error(Y_test,rf_predictions)

print(f"Mean squared error: ${rf_mse:.2f}")
print(f"Mean absolute error: ${rf_mae:.2f}")
      

      






