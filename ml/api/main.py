# ml/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # For defining request body schema
import uvicorn
import os
import pandas as pd
import psycopg2
import numpy as np
import pickle
import mlflow
import json # To parse Coordinate string if needed

# Import necessary components from train_model.py
from ml.scripts.train_model import get_db_connection, perform_feature_engineering, get_hybrid_recommendations # <--- IMPORTANT

# --- MLflow Configuration ---
# Set MLflow tracking URI (points to your local MLflow server)
# In production, this would be a remote MLflow server (e.g., on your VM)
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI', 'http://host.docker.internal:5000')

# --- Global Variables for Loaded Models and Data ---
# These will be loaded once when the API starts
cf_model = None
tfidf_vectorizer = None
ohe_encoder = None
places_df = None # Full places_df for getting details
content_features_df = None # Engineered content features for CB
ratings_df_raw = None # Raw ratings for get_hybrid_recommendations

# --- Request Body Schema ---
class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = 10

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Indonesian Tourism Recommender ML Inference API",
    description="API for generating hybrid tourism recommendations using ML models.",
    version="1.0.0"
)

# --- Startup Event: Load Models and Data ---
@app.on_event("startup")
async def load_models_and_data():
    global cf_model, tfidf_vectorizer, ohe_encoder, places_df, content_features_df, ratings_df_raw

    print("--- ML Inference API Startup: Loading models and data ---")

    try:
        # 1. Load data from DB (we need latest places and ratings for inference)
        conn = get_db_connection()
        if conn:
            places_df, users_df_raw, ratings_df_raw = perform_initial_data_load_for_api(conn)
            conn.close()
        else:
            raise Exception("Failed to connect to database for initial data load.")
        
        # 2. Re-perform Feature Engineering to get TF-IDF and OHE on full data
        # Note: In a real MLOps pipeline, these might be loaded as pre-computed features or models.
        # For simplicity, we re-run FE to get content_features_df based on current places_df.
        content_features_df, tfidf_vectorizer, ohe_encoder = perform_feature_engineering(
            places_df.copy(), users_df_raw.copy(), ratings_df_raw.copy()
        )
        # Ensure 'combined_text' is added to places_df for potential re-use if needed
        places_df['combined_text'] = places_df['place_name'] + " " + places_df['description']

        # 3. Load MLflow Artifacts (CF Model, TF-IDF, OHE)
        # Assuming you logged them in the latest run or a specific production run.
        # For this example, we'll pick the latest successful run for 'SVD_CF_Model_Training'
        # In production, you'd specify a 'production' model from MLflow Model Registry
        
        # A more robust way: use MLflow Model Registry to load a specific version
        # e.g., model_name = "SVDCollaborativeFilteringModel"
        # client = mlflow.tracking.MlflowClient()
        # model_version = client.get_latest_versions(model_name, stages=["Production"])[0].version
        # cf_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

        # For simplicity, let's load artifacts directly from file system as they are local
        # This means the MLflow artifacts must be available in the container's path,
        # which they will be if we build the container after training and copying mlruns.

        # Find the latest run's artifacts path
        # You'll need to manually find your experiment ID and latest run ID here,
        # or use MLflowClient to programmatically find the latest run with status 'FINISHED'.
        # For now, let's assume artifacts are in a known place relative to this script after Docker build.
        
        # Simpler approach: directly load the pickle files that were saved in mlruns during training.
        # This requires copying the mlruns directory into the Docker image, which we'll do.
        print("Attempting to load models from MLflow artifacts...")
        
        # You'll need to find the correct path to the latest run ID's artifacts.
        # Let's mock this for now, assuming artifacts are copied to /app/mlruns/<exp_id>/<run_id>/artifacts/
        # or we could make train_model.py save them to a fixed shared 'models/' directory
        # Let's try loading from a fixed 'models/' path that we'll copy into the Dockerfile.
        # Train_model.py needs to save them to this fixed path: ml/models/
        
        # TEMP: For development, let's reload them from the *local* mlruns directory's latest run artifact.
        # This is not robust for production as the specific run_id changes.
        # A better way for production: Use MlflowClient or ensure train_model saves to a fixed path.
        
        # For now, let's simply assume train_model.py directly saves them to ml/models/
        # So we load from there. This is easier for Dockerfile.
        
        # Re-using the initial load method for API.
        # The trained_model.py already saves to `cf_model.pkl`, `tfidf_vectorizer.pkl`, `ohe_encoder.pkl`
        # in the working directory from where the script runs.
        # We need to make sure these are accessible to the API.
        # Let's ensure train_model.py saves them to `ml/models/` explicitly.
        
        # This implies a change in train_model.py first: save to ml/models/ folder.
        # I will update train_model.py to save to ml/models/
        
        # Assuming models are saved to ml/models/
        with open("ml/models/cf_model.pkl", "rb") as f:
            cf_model = pickle.load(f)
        with open("ml/models/tfidf_vectorizer.pkl", "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        with open("ml/models/ohe_encoder.pkl", "rb") as f:
            ohe_encoder = pickle.load(f)
        print("Models and feature engineering tools loaded successfully.")

    except Exception as e:
        print(f"Error during API startup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load models or data: {e}")

# Re-implementing initial data load for API startup
# This avoids global variables from train_model.py directly
def perform_initial_data_load_for_api(conn):
    df_places, df_users, df_ratings = None, None, None
    try:
        # Load from DB (similar to train_model.py's load_data_from_db)
        df_places = pd.read_sql("SELECT * FROM places;", conn)
        df_users = pd.read_sql("SELECT * FROM users;", conn)
        df_ratings = pd.read_sql("SELECT * FROM ratings;", conn)

        # Rename columns to match what get_hybrid_recommendations expects (lowercase)
        df_places.rename(columns={
            'place_id': 'place_id', 'place_name': 'place_name', 'description': 'description',
            'category': 'category', 'city': 'city', 'price': 'price', 'rating': 'rating',
            'time_minutes': 'time_minutes', 'coordinate': 'coordinate', 'lat': 'lat', 'long': 'long'
        }, inplace=True)
        df_users.rename(columns={
            'user_id': 'user_id', 'location': 'location', 'age': 'age'
        }, inplace=True)
        df_ratings.rename(columns={
            'user_id': 'user_id', 'place_id': 'place_id', 'place_rating': 'place_rating'
        }, inplace=True)
        df_ratings['user_id'] = df_ratings['user_id'].astype(int)
        df_ratings['place_id'] = df_ratings['place_id'].astype(int)
        df_ratings['place_rating'] = df_ratings['place_rating'].astype(int)

    except Exception as e:
        print(f"Error loading initial data for API: {e}")
        raise
    return df_places, df_users, df_ratings

# --- Health Check Endpoint ---
@app.get("/health", summary="Health Check", tags=["Monitoring"])
async def health_check():
    """Checks if the ML inference API is running and models are loaded."""
    if cf_model is not None and tfidf_vectorizer is not None and ohe_encoder is not None:
        return {"status": "ok", "message": "ML Inference API is healthy and models loaded."}
    return {"status": "error", "message": "ML Inference API is running but models not loaded."}

# --- Recommendation Endpoint ---
@app.post("/recommend", summary="Get Hybrid Recommendations", response_model=list[dict], tags=["Recommendations"])
async def get_recommendations_api(request: RecommendationRequest):
    """
    Generates a list of hybrid tourism destination recommendations for a given user.
    """
    if cf_model is None or tfidf_vectorizer is None or ohe_encoder is None or places_df is None or content_features_df is None or ratings_df_raw is None:
        raise HTTPException(status_code=503, detail="ML models or data not loaded. Please wait for startup or check logs.")

    try:
        # Call the hybrid recommendation logic
        # We need to ensure get_hybrid_recommendations is available.
        # We've imported it from ml.scripts.train_model, but it depends on other global objects.
        # It's better to pass all necessary data or make get_hybrid_recommendations a class method.
        # For simplicity, let's make sure get_hybrid_recommendations directly uses the global vars passed by context or is part of a class.
        
        # The get_hybrid_recommendations function expects:
        # user_id, cf_model, places_df, content_features_df, tfidf_vectorizer, ohe_encoder, ratings_df
        
        # Let's ensure the function in train_model.py is self-contained or passed everything.
        # For now, we'll rely on the global variables being set by startup.
        
        # The `get_hybrid_recommendations` function uses DB connection inside. This is problematic for an API.
        # The API should NOT open and close DB connection per request.
        # We need to refactor `get_hybrid_recommendations` to receive `rated_place_ids` directly,
        # or have `places_df` and `ratings_df` pre-loaded and passed.
        # We've already passed `ratings_df_raw` globally.
        
        # Let's modify `get_hybrid_recommendations` to take `rated_place_ids` as an explicit argument if we want it to be stateless for API.
        # Or, we make it aware of the pre-loaded global `ratings_df_raw`.
        
        # For now, let's keep get_hybrid_recommendations as is, and make sure it uses the global ratings_df_raw
        # The current get_hybrid_recommendations already loads user ratings via DB. This is not good for API performance.
        # We need to change get_hybrid_recommendations to take `user_ratings_for_cf` from outside.
        # Or, the API will query DB for user ratings for that specific user.

        # For the API, we will load `places_df` and `ratings_df_raw` once at startup.
        # The `get_hybrid_recommendations` needs to be adapted:
        # It takes `user_id`, `cf_model`, `places_df`, `content_features_df`, `tfidf_vectorizer`, `ohe_encoder`,
        # but also implicitly needs `ratings_df_raw` (or a specific user's ratings).

        # Refactoring `get_hybrid_recommendations` slightly for API usage:
        # It should take `user_ratings_df` (dataframe of ratings for this user) as an argument.
        # The API will query the database for this specific user's ratings.

        # --- Refactor get_hybrid_recommendations for API ---
        # The current get_hybrid_recommendations loads user_ratings_df from DB.
        # This should be done by API outside the recommendation logic for performance.
        conn_api = get_db_connection()
        if not conn_api:
            raise HTTPException(status_code=500, detail="Database connection failed for user ratings.")
        try:
            user_ratings_df_for_request = pd.read_sql(f"SELECT place_id, place_rating FROM ratings WHERE user_id = {request.user_id};", conn_api)
        finally:
            conn_api.close()

        # Pass all necessary pre-loaded data and specific user ratings
        recommendations_df = get_hybrid_recommendations(
            user_id=request.user_id,
            cf_model=cf_model,
            places_df=places_df,
            content_features_df=content_features_df,
            tfidf_vectorizer=tfidf_vectorizer,
            ohe_encoder=ohe_encoder,
            ratings_df=ratings_df_raw, # Pass the full pre-loaded ratings_df_raw
            num_recommendations=request.num_recommendations
        )

        if recommendations_df.empty:
            return []
        
        # Convert DataFrame to list of dictionaries for JSON response
        # Ensure 'coordinate' is serializable (it's a string, so fine)
        # Convert float32 to float for JSON compatibility
        recommendations_df = recommendations_df.astype({'lat': float, 'long': float, 'rating': float, 'price': float, 'time_minutes': float})

        # Drop the 'combined_text' column if it's not needed in API response
        if 'combined_text' in recommendations_df.columns:
            recommendations_df = recommendations_df.drop(columns=['combined_text'])


        return recommendations_df.to_dict(orient="records")

    except Exception as e:
        print(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- To run this API (for local development only, Uvicorn will be used in Docker) ---
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)