import pandas as pd
import joblib

class ProductRecommender:
    def __init__(self,model_path="rf_model.joblib",scaler_path="scaler.joblib"):
        self.model=joblib.load(model_path)
        self.scaler=joblib.load(scaler_path)

    def prepare_features(self,customer_id,all_products_df,base_features_df):
        customer_data=base_features_df[base_features_df["customer_id"] == customer_id].copy()
        if customer_data.empty:
            raise ValueError(f"Customer ID {customer_id} için veri bulunamadı.") 

        product_combinations = all_products_df[["product_id"]].drop_duplicates().copy()
        for col in customer_data.columns:
            if col not in ["product_id"]:
                product_combinations[col]=customer_data.iloc[0][col]   
        
        scaled_features=self.scaler.transform(product_combinations)
        return product_combinations["product_id"],scaled_features
    
    def recommend(self,customer_id,all_products_df,base_features_df,n=5):
        product_ids,features=self.prepare_features(customer_id,all_products_df,base_features_df)
        predictions = self.model.predict(features)

        quantity_preds=predictions[:,1]
        recommendation_df=pd.DataFrame({
            "product_id":product_ids,
            "predicted_quantity":quantity_preds
        }).sort_values(by="predicted_quantity",ascending=False).head(n)

        return recommendation_df.reset_index(drop=True)

