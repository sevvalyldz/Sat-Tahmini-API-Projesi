import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import time
from getdatabase import GetDatabase
from productrecommender import ProductRecommender



db = GetDatabase(
    username="postgres",
    password="password",
    host="localhost",
    port="5432",
    database="GYK2Northwind"
)

orders_df = db.fetch_data("orders")
products_df = db.fetch_data("products")
order_details_df = db.fetch_data("Order_Details")
customers_df = db.fetch_data("Customers")
monthly_sales_df = db.fetch_data("monthly_sales ")
customer_sales_df = db.fetch_data("customer_sales ")
categories_df = db.fetch_data("categories")

def train_and_evaluate_models(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    base_models = {
        "KNN": KNeighborsRegressor(),
        "Karar Ağacı": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Lineer Regresyon": LinearRegression(),
        "SVR": SVR()
    }

    results = []

    for name, base_model in base_models.items():
        model = MultiOutputRegressor(base_model)
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_time = time.time()

        model_results = {"Model": name, "Training Time (s)": round(end_time - start_time, 4)}

        for i, col in enumerate(y.columns):
            model_results[f"{col} MSE"] = round(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]), 4)
            model_results[f"{col} MAE"] = round(mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]), 4)
            model_results[f"{col} R2"] = round(r2_score(y_test.iloc[:, i], y_pred[:, i]), 4)

        results.append(model_results)

    return pd.DataFrame(results)

order_data = pd.merge(
    order_details_df,
    orders_df[["order_id", "customer_id", "order_date"]],
    on="order_id",
    how="left"
)

order_data["Month"] = pd.to_datetime(order_data["order_date"]).dt.month
order_data["Year"] = pd.to_datetime(order_data["order_date"]).dt.year

order_data = pd.merge(
    order_data,
    customer_sales_df,
    on="customer_id",
    how="left"
)

order_data = pd.merge(
    order_data,
    products_df[["product_id", "product_name", "category_id"]],
    on="product_id",
    how="left"
)

order_data = pd.merge(
    order_data,
    categories_df[["category_id"]],
    on="category_id",
    how="left"
)

order_data = pd.merge(
    order_data,
    customers_df[["customer_id"]],
    on="customer_id",
    how="left"
)

order_data["TotalPrice"] = order_data["unit_price"] * order_data["quantity"]
order_data["Segment"] = order_data["Segment"].astype("category").cat.codes
order_data["has_discount"] = (order_data["discount"] > 0).astype(int)

q_high = order_data["quantity"].quantile(0.99)
order_data = order_data[order_data["quantity"] < q_high]

order_count = orders_df.groupby("customer_id").size().reset_index(name="order_count")
avg_spending = order_data.groupby("customer_id")["TotalPrice"].mean().reset_index(name="avg_spending")
order_data = pd.merge(order_data, order_count, on="customer_id", how="left")
order_data = pd.merge(order_data, avg_spending, on="customer_id", how="left")

monthly_product_sales = (
    order_data.groupby(["product_id", "Year", "Month"])["quantity"]
    .sum()
    .reset_index()
    .sort_values(by=["product_id", "Year", "Month"])
)
monthly_product_sales["previous_quantity"] = monthly_product_sales.groupby("product_id")["quantity"].shift(1)

order_data = pd.merge(
    order_data,
    monthly_product_sales[["product_id", "Year", "Month", "previous_quantity"]],
    on=["product_id", "Year", "Month"],
    how="left"
)

order_data["previous_quantity"] = order_data["previous_quantity"].fillna(0)
order_data["category_id"] = order_data["category_id"].astype("category").cat.codes
avg_price_product = order_data.groupby("product_id")["unit_price"].mean().reset_index(name="product_avg_price")
order_data = pd.merge(order_data, avg_price_product, on="product_id", how="left")



X = order_data[[
    "product_id", "unit_price", "Month", "customer_id", "Segment",
    "has_discount", "order_count", "avg_spending", "previous_quantity", "category_id"
]]
X["customer_id"] = X["customer_id"].astype("category").cat.codes

y = order_data[["TotalPrice", "quantity"]]
db.create_data(X, "order_details_model_data")

results = train_and_evaluate_models(X, y)
print(results)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_scaled, y)


joblib.dump(model, "rf_model.joblib")
joblib.dump(scaler, "scaler.joblib")

all_products_df = X.drop_duplicates(subset=["product_id"])[["product_id"]].copy()
base_features_df = X.copy()

recommender = ProductRecommender()
recommendations=recommender.recommend(customer_id=5, all_products_df=all_products_df, base_features_df=base_features_df)
recommendations = recommendations.merge(products_df[["product_id", "product_name"]], on="product_id", how="left")
recommendations["score"] = recommendations["predicted_quantity"] / recommendations["predicted_quantity"].max()
print(recommendations)
