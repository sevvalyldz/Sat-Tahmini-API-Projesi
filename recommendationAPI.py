from fastapi import FastAPI,Query,HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from getdatabase import GetDatabase
from productrecommender import ProductRecommender
import joblib

app=FastAPI()

class PredictionInput(BaseModel):
    product_id: int
    unit_price: float
    Month: int
    customer_id: int
    Segment: int
    has_discount: int
    order_count: int
    avg_spending: float
    previous_quantity: float
    category_id: int

class RecommendationAPI:
    def __init__(self):
        self.model=joblib.load("rf_model.joblib")
        self.scaler=joblib.load("scaler.joblib")
        self.recommender= ProductRecommender()
        
        db = GetDatabase(
            username="postgres",
            password="password",
            host="localhost",
            port="5432",
            database="GYK2Northwind"
        )
        order_data=db.fetch_data("order_details_model_data")
        order_data=order_data.dropna(subset=["product_id","customer_id"])
        products_df = db.fetch_data("products")[["product_id", "product_name"]]
        self.X=order_data.copy() 
        self.base_features_df=self.X.copy()
        self.X = self.X.merge(products_df, on="product_id", how="left")
        self.all_products_df=self.X.drop_duplicates(subset=["product_id"])[["product_id","product_name"]].copy()
       

    def recommend_products(self, customer_id: int, top_n: int = 5):
        try:    
            recommendations=self.recommender.recommend(
                customer_id=customer_id,
                all_products_df=self.all_products_df,
                base_features_df=self.base_features_df,
                n=top_n
            )
            recommendations = recommendations.merge(
            self.all_products_df, on="product_id", how="left"
        )
            recommendations = recommendations[["product_id", "product_name", "predicted_quantity"]]
            
            return recommendations.to_dict(orient="records")
        except ValueError as e:
            raise HTTPException(status_code=404,detail=str(e))
        
    def predict_sales(self, input_data: PredictionInput):
        input_df = pd.DataFrame([input_data.dict()])
        input_scaled = self.scaler.transform(input_df)
        prediction = self.model.predict(input_scaled)

        return {
            "TotalPrice": round(prediction[0][0], 2),
            "Quantity": round(prediction[0][1], 2)
        }    
    def retrain_model(self):
        db = GetDatabase(
        username="postgres",
        password="password",
        host="localhost",
        port="5432",
        database="GYK2Northwind"
    )
        order_data = db.fetch_data("order_details_model_data").dropna()
        order_details = db.fetch_data("order_details")[["order_id", "product_id", "quantity"]]
        order_data = pd.merge(order_data, order_details, on=["product_id"], how="left")
        order_data["TotalPrice"] = order_data["unit_price"] * order_data["quantity"]
        


        X = order_data[["product_id", "unit_price", "Month", "customer_id", "Segment",
        "has_discount", "order_count", "avg_spending", "previous_quantity", "category_id"]]
        y = order_data[["TotalPrice", "quantity"]]

        X["customer_id"] = X["customer_id"].astype("category").cat.codes
        X["Segment"] = X["Segment"].astype("category").cat.codes

   
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
        model.fit(X_train_scaled, y_train)

    
        joblib.dump(model, "rf_model.joblib")
        joblib.dump(scaler, "scaler.joblib")

        return {"message": "Model başarıyla yeniden eğitildi."}
    
    def get_sales_summary(self):
        db = GetDatabase(
            username="postgres",
            password="password",
            host="localhost",
            port="5432",
            database="GYK2Northwind"
        )
        monthly_sales=db.fetch_data("monthly_sales")
        customer_sales=db.fetch_data("customer_sales")
        return {
            "monthly_sales":monthly_sales.to_dict(orient="records"),
            "customer_sales":customer_sales.to_dict(orient="records")
        }

        
api=RecommendationAPI()

@app.get("/products")
def get_products():
    return api.all_products_df[["product_id","product_name"]].drop_duplicates().to_dict(orient="records")

@app.get("/recommend")
def recommend(customer_id: int= Query(...),top_n: int=Query(5)):
    return api.recommend_products(customer_id,top_n)

@app.post("/predict")
def predict(input_data: PredictionInput):
    return api.predict_sales(input_data)

@app.post("/retrain")
def retrain():
    return api.retrain_model()

@app.get("/sales_summary")
def sales_summary():
    return api.get_sales_summary()