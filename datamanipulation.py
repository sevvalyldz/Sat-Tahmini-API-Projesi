from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from getdatabase import GetDatabase 

# GetDatabase sınıfını başlatırken gerekli parametreleri veriyoruz
db = GetDatabase(
    username="postgres",
    password="password",
    host="localhost",
    port="5432",
    database="GYK2Northwind"
)

# Veri çekme işlemi
orders_df = db.fetch_data("orders")
print("Orders Tablosu:")
print(orders_df.head())

products_df = db.fetch_data("products")
print("Products Tablosu:")
print(products_df.head())

order_details_df = db.fetch_data("Order_Details")
print("Order Details Tablosu:")
print(order_details_df.head())

customers_df = db.fetch_data("Customers")
print("Customers Tablosu:")
print(customers_df.head())

categories_df = db.fetch_data("Categories ")
print("Categories Tablosu:")
print(categories_df.head())

merged_df = pd.merge(order_details_df, orders_df, on="order_id", how="inner")
merged_df = pd.merge(merged_df, products_df, on="product_id", how="inner")


merged_df['order_date'] = pd.to_datetime(
    merged_df['order_date'])  
merged_df['Month'] = merged_df['order_date'].dt.to_period('M')


monthly_sales = merged_df.groupby(['Month', 'product_id']).agg({"quantity": "sum"}).reset_index()
monthly_sales['Month'] = monthly_sales['Month'].astype(str)
print(monthly_sales.head())

merged_df['TotalPrice'] = merged_df['quantity'] * merged_df['unit_price_y']
customer_sales = merged_df.groupby("customer_id")["TotalPrice"].sum()
customer_sales_segmented = pd.cut(customer_sales, bins=[0, 1000, 5000, 10000, np.inf], labels=["Low", "Medium", "High", "VIP"])
customer_sales = customer_sales.to_frame(name="TotalPrice")
customer_sales["Segment"] = customer_sales_segmented
customer_sales = customer_sales.reset_index()
print(customer_sales)

db.create_data(monthly_sales,"monthly_sales")
db.create_data(customer_sales,"customer_sales")
