# ml/scripts/train_model.py

import pandas as pd
import psycopg2
import os
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import SVD
from surprise import accuracy

import mlflow
# import mlflow.surprise # <--- REMOVED this line as it doesn't exist
import pickle # <--- NEW: For saving model as .pkl


# Register numpy.int64 adapter for psycopg2
register_adapter(np.int64, AsIs)


# Database connection details
DB_HOST = os.getenv('DB_HOST', 'host.docker.internal')
DB_NAME = os.getenv('DB_NAME', 'deploycamp')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '1')
DB_PORT = os.getenv('DB_PORT', '5435')

def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("Database connection successful.")
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def load_data_from_db(conn):
    """Loads data from PostgreSQL tables into Pandas DataFrames and renames columns."""
    if conn is None:
        return None, None, None

    df_places = None
    df_users = None
    df_ratings = None

    try:
        print("Loading data from 'places' table...")
        df_places = pd.read_sql("SELECT * FROM places;", conn)
        df_places.rename(columns={
            'place_id': 'place_id',
            'place_name': 'place_name',
            'description': 'description',
            'category': 'category',
            'city': 'city',
            'price': 'price',
            'rating': 'rating',
            'time_minutes': 'time_minutes',
            'coordinate': 'coordinate',
            'lat': 'lat',
            'long': 'long'
        }, inplace=True)
        print(f"Loaded {len(df_places)} rows from 'places'.")

        print("Loading data from 'users' table...")
        df_users = pd.read_sql("SELECT * FROM users;", conn)
        df_users.rename(columns={
            'user_id': 'user_id',
            'location': 'location',
            'age': 'age'
        }, inplace=True)
        print(f"Loaded {len(df_users)} rows from 'users'.")

        print("Loading data from 'ratings' table...")
        df_ratings = pd.read_sql("SELECT * FROM ratings;", conn)
        df_ratings.rename(columns={
            'user_id': 'user_id',
            'place_id': 'place_id',
            'place_rating': 'place_rating'
        }, inplace=True)
        df_ratings['user_id'] = df_ratings['user_id'].astype(int)
        df_ratings['place_id'] = df_ratings['place_id'].astype(int)
        df_ratings['place_rating'] = df_ratings['place_rating'].astype(int)
        print(f"Loaded {len(df_ratings)} rows from 'ratings'.")

    except Exception as e:
        print(f"Error loading data from database: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")
    
    return df_places, df_users, df_ratings

def perform_feature_engineering(places_df, users_df, ratings_df):
    """Performs feature engineering steps."""
    print("\n--- Starting Feature Engineering ---")

    places_df['combined_text'] = places_df['place_name'] + " " + places_df['description']
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=3)
    tfidf_matrix = tfidf_vectorizer.fit_transform(places_df['combined_text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=places_df.index)
    
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    categorical_cols = ['category', 'city']
    encoded_features = ohe.fit_transform(places_df[categorical_cols])
    encoded_feature_names = ohe.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=places_df.index)
    
    numerical_cols = ['price', 'rating', 'time_minutes', 'lat', 'long']
    numerical_df = places_df[numerical_cols].copy()

    content_features_df = pd.concat([tfidf_df, encoded_df, numerical_df], axis=1)
    print(f"Combined Content-Based Features Shape: {content_features_df.shape}")

    print("\n--- Feature Engineering Completed ---")
    return content_features_df, tfidf_vectorizer, ohe # Return models for later use

def split_data(places_df, content_features_df, ratings_df):
    """Splits data for Content-Based and Collaborative Filtering models."""
    print("\n--- Starting Data Splitting ---")

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'place_id', 'place_rating']], reader)
    
    trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)
    
    # print(f"Collaborative Filtering Trainset Size: {len(trainset.all_ratings())} interactions")
    print(f"Collaborative Filtering Testset Size: {len(testset)} interactions")

    print("\n--- Data Splitting Completed ---")
    return trainset, testset, places_df, content_features_df

def train_models(trainset, testset, content_features_df, tfidf_vectorizer, ohe_encoder): # <--- NEW: added tfidf_vectorizer, ohe_encoder
    """Trains the Collaborative Filtering model and evaluates it, logging to MLflow."""
    print("\n--- Starting Model Training ---")

    # MLflow tracking
    mlflow.set_experiment("Tourism Recommendation System") # <--- NEW: Explicitly set experiment
    with mlflow.start_run(run_name="SVD_CF_Model_Training"):
        # Log parameters
        mlflow.log_param("cf_model_type", "SVD")
        mlflow.log_param("surprise_test_size", 0.2)
        mlflow.log_param("surprise_random_state", 42)
        
        # --- Train Collaborative Filtering Model (SVD) ---
        print("Training Collaborative Filtering (SVD) model...")
        cf_model = SVD(random_state=42)
        cf_model.fit(trainset)
        print("Collaborative Filtering (SVD) model trained.")

        # --- Evaluate Collaborative Filtering Model ---
        print("Evaluating Collaborative Filtering model...")
        predictions = cf_model.test(testset)
        
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        
        print(f"Collaborative Filtering Model RMSE: {rmse}")
        print(f"Collaborative Filtering Model MAE: {mae}")

        # Log metrics to MLflow
        mlflow.log_metric("cf_rmse", rmse)
        mlflow.log_metric("cf_mae", mae)

        # Log the trained CF model as a pickle artifact
        print("Logging CF model to MLflow...")
        # Define artifact path within MLflow run
        cf_model_path = "cf_model.pkl"
        with open(cf_model_path, "wb") as f:
            pickle.dump(cf_model, f)
        mlflow.log_artifact(cf_model_path, "cf_model_artifact") # <--- NEW: Log as artifact

        # Log the TF-IDF Vectorizer and OneHotEncoder as artifacts
        print("Logging TF-IDF Vectorizer and OneHotEncoder to MLflow...")
        tfidf_path = "tfidf_vectorizer.pkl"
        ohe_path = "ohe_encoder.pkl"
        with open(tfidf_path, "wb") as f:
            pickle.dump(tfidf_vectorizer, f)
        with open(ohe_path, "wb") as f:
            pickle.dump(ohe_encoder, f)
        
        mlflow.log_artifact(tfidf_path, "feature_engineering_artifacts")
        mlflow.log_artifact(ohe_path, "feature_engineering_artifacts")
        print("Feature engineering tools logged to MLflow.")

        # --- Content-Based Model Preparation ---
        print("Content-Based model: Features prepared. Similarity calculation will be done during inference.")
        
        print("\n--- Model Training & Evaluation Completed ---")
        return cf_model # Return the trained CF model


if __name__ == "__main__":
    conn = get_db_connection()
    if conn:
        places_df_raw, users_df_raw, ratings_df_raw = load_data_from_db(conn)

        # Pass tfidf_vectorizer and ohe_encoder from feature engineering to train_models
        content_features_df, tfidf_vectorizer, ohe_encoder = perform_feature_engineering(
            places_df_raw.copy(), users_df_raw.copy(), ratings_df_raw.copy()
        )

        cf_trainset, cf_testset, full_places_df, full_content_features_df = split_data(
            places_df_raw, content_features_df, ratings_df_raw.copy()
        )

        # Train and evaluate models (CF model for now)
        trained_cf_model = train_models(
            cf_trainset, 
            cf_testset, 
            full_content_features_df,
            tfidf_vectorizer, # <--- NEW: Pass tfidf_vectorizer
            ohe_encoder       # <--- NEW: Pass ohe_encoder
        )

        print("\nModel training script execution complete.")
        print("Check MLflow UI at http://localhost:5000 to see experiment runs.")