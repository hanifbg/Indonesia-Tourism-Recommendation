# ml/scripts/train_model.py

import pandas as pd
import psycopg2
import os
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import SVD
from surprise import accuracy

import mlflow
import pickle

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
    
    print(f"Collaborative Filtering Trainset Size: {trainset.n_ratings} interactions")
    print(f"Collaborative Filtering Testset Size: {len(testset)} interactions")

    print("\n--- Data Splitting Completed ---")
    return trainset, testset, places_df, content_features_df

def train_models(trainset, testset, content_features_df, tfidf_vectorizer, ohe_encoder):
    """Trains the Collaborative Filtering model and evaluates it, logging to MLflow."""
    print("\n--- Starting Model Training ---")

    mlflow.set_experiment("Tourism Recommendation System")
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
        cf_model_path = "cf_model.pkl"
        with open(cf_model_path, "wb") as f:
            pickle.dump(cf_model, f)
        mlflow.log_artifact(cf_model_path, "cf_model_artifact")

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

# --- NEW: Hybrid Recommendation Logic Function ---
# FIX: Added ratings_df to the function arguments
def get_hybrid_recommendations(
    user_id: int, 
    cf_model, 
    places_df: pd.DataFrame, 
    content_features_df: pd.DataFrame, 
    tfidf_vectorizer: TfidfVectorizer, 
    ohe_encoder: OneHotEncoder, 
    ratings_df: pd.DataFrame, # <--- NEW: Add ratings_df here
    num_recommendations: int = 10,
    top_n_cf: int = 20,
    top_n_cb: int = 20,
    cf_weight: float = 0.7
) -> pd.DataFrame:
    """
    Generates hybrid recommendations for a given user ID.
    Combines Collaborative Filtering and Content-Based approaches.
    Handles cold-start users (no ratings) by falling back to content-based or popularity.
    """
    print(f"\nGenerating hybrid recommendations for user_id: {user_id}")

    # 1. Collaborative Filtering (CF) Recommendations
    # Get a list of all place_ids the user has NOT rated yet
    # Use the passed ratings_df instead of trying to get it from DB again
    user_ratings_for_cf = ratings_df[ratings_df['user_id'] == user_id]
    rated_place_ids = user_ratings_for_cf['place_id'].tolist()
    
    # Get unrated places
    unrated_places = places_df[~places_df['place_id'].isin(rated_place_ids)]
    
    if len(user_ratings_for_cf) > 0: # User has ratings - use CF
        print(f"User {user_id} has {len(user_ratings_for_cf)} existing ratings. Using CF.")
        cf_predictions = []
        for place_id in unrated_places['place_id']:
            # Predict rating for each unrated place
            # Ensure the place_id is known to the CF model (may not be for brand new places)
            if cf_model.trainset.knows_item(place_id):
                predicted_rating = cf_model.predict(user_id, place_id).est
                cf_predictions.append({'place_id': place_id, 'predicted_rating': predicted_rating})
            # else: skip this place if CF model doesn't know it, rely on CB
        
        cf_recs_df = pd.DataFrame(cf_predictions).sort_values(by='predicted_rating', ascending=False)
        
    else: # Cold-start user - no ratings
        print(f"User {user_id} is cold-start (no ratings). Relying on Content-Based and popularity.")
        cf_recs_df = pd.DataFrame() # Empty CF recs
        cf_weight = 0.0 # No CF contribution for cold-start

    # 2. Content-Based (CB) Recommendations (for cold-start or to diversify CF)
    cb_recs_df = pd.DataFrame()
    
    if len(user_ratings_for_cf) > 0: # User has ratings - find content-similar to their liked places
        # Find user's top-rated places (e.g., top 3) to use as content base
        top_user_rated_places_ids = user_ratings_for_cf.sort_values(by='place_rating', ascending=False).head(3)['place_id'].tolist()
        
        if top_user_rated_places_ids: # Ensure there are top rated places to use for CB
            # Get content features of these top-rated places
            # Ensure indexing matches place_id correctly. Assuming place_id is 1-indexed.
            # Convert place_id to 0-indexed for DataFrame lookup if necessary: places_df.index is 0-indexed.
            # We need to map place_id to the index of content_features_df.
            
            # Create a mapping from place_id to content_features_df index
            place_id_to_index = {pid: idx for idx, pid in enumerate(places_df['place_id'])}
            
            top_rated_place_indices = [place_id_to_index[pid] for pid in top_user_rated_places_ids if pid in place_id_to_index]
            
            if top_rated_place_indices:
                top_rated_place_content_features = content_features_df.iloc[top_rated_place_indices]
                
                all_content_features_np = content_features_df.to_numpy() # Convert once
                top_rated_place_content_features_np = top_rated_place_content_features.to_numpy()

                similarity_scores = cosine_similarity(top_rated_place_content_features_np, all_content_features_np)
                avg_similarity_scores = np.mean(similarity_scores, axis=0)
                
                # Get place_ids sorted by similarity, excluding rated ones
                # Get the actual place_id from the original places_df based on index
                similar_place_idx_series = pd.Series(avg_similarity_scores, index=places_df['place_id']).sort_values(ascending=False)
                
                # Filter out already rated places and take top_n
                cb_recs_series = similar_place_idx_series[~similar_place_idx_series.index.isin(rated_place_ids)].head(top_n_cb)
                
                cb_recs_df = pd.DataFrame({
                    'place_id': cb_recs_series.index,
                    'cb_score': cb_recs_series.values
                })
        else: # No top rated places found for CB, e.g. if user only rated 1-star
            print("No sufficiently top-rated places found for content-based recommendations for this user.")
            cb_recs_df = pd.DataFrame() # Empty CB recs
            
    else: # Cold-start, fallback to top-rated places overall (popularity)
        print("Cold-start user: Recommending top-rated places overall for content-based fallback.")
        cb_recs_df = places_df.sort_values(by='rating', ascending=False).head(top_n_cb).copy()
        cb_recs_df['cb_score'] = 1.0 # Assign a high score for blending, indicating high confidence in popularity
        cb_recs_df = cb_recs_df[~cb_recs_df['place_id'].isin(rated_place_ids)] # Ensure no rated places for cold-start (though shouldn't be anyway)


    # 3. Hybridization
    final_recommendations = pd.DataFrame()
    
    # Merge CF and CB recommendations if both exist
    if not cf_recs_df.empty and not cb_recs_df.empty:
        # Perform a full outer join to combine all recommendations
        hybrid_df = pd.merge(cf_recs_df[['place_id', 'predicted_rating']], 
                             cb_recs_df[['place_id', 'cb_score']], 
                             on='place_id', how='outer')
        
        # Fill NaN scores with 0 if a place was only recommended by one method
        hybrid_df['predicted_rating'] = hybrid_df['predicted_rating'].fillna(0)
        hybrid_df['cb_score'] = hybrid_df['cb_score'].fillna(0)

        # Normalize CF predicted_rating to a 0-1 scale to blend with CB score
        min_cf_rating, max_cf_rating = 1, 5 # Ratings are 1-5
        hybrid_df['predicted_rating_normalized'] = (hybrid_df['predicted_rating'] - min_cf_rating) / (max_cf_rating - min_cf_rating)
        
        # Calculate hybrid score
        hybrid_df['hybrid_score'] = (cf_weight * hybrid_df['predicted_rating_normalized']) + \
                                    ((1 - cf_weight) * hybrid_df['cb_score'])
        
        # Sort by hybrid score, then by original place rating (as a tie-breaker or fallback quality)
        final_recommendations = hybrid_df.sort_values(by='hybrid_score', ascending=False)
        
    elif not cf_recs_df.empty: # Only CF recommendations available
        final_recommendations = cf_recs_df.sort_values(by='predicted_rating', ascending=False)
    elif not cb_recs_df.empty: # Only CB recommendations available (e.g., cold-start)
        final_recommendations = cb_recs_df.sort_values(by='cb_score', ascending=False)
    else:
        print("No recommendations found for this user.")
        return pd.DataFrame() # Return empty DataFrame

    # Filter out already rated places (again, to be safe, as CB/CF might include them before final merge)
    final_recommendations = final_recommendations[~final_recommendations['place_id'].isin(rated_place_ids)]
    
    # Merge with original places_df to get full details of recommended places
    recommended_places_details = pd.merge(final_recommendations[['place_id']].head(num_recommendations), places_df, on='place_id', how='left')

    return recommended_places_details.head(num_recommendations)


if __name__ == "__main__":
    conn = get_db_connection()
    if conn:
        places_df_raw, users_df_raw, ratings_df_raw = load_data_from_db(conn)

        content_features_df, tfidf_vectorizer, ohe_encoder = perform_feature_engineering(
            places_df_raw.copy(), users_df_raw.copy(), ratings_df_raw.copy()
        )

        cf_trainset, cf_testset, full_places_df, full_content_features_df = split_data(
            places_df_raw, content_features_df, ratings_df_raw.copy()
        )

        trained_cf_model = train_models(
            cf_trainset, 
            cf_testset, 
            full_content_features_df,
            tfidf_vectorizer,
            ohe_encoder
        )

        print("\nModel training script execution complete.")
        print("Check MLflow UI at http://localhost:5000 to see experiment runs.")

        # --- NEW: Test Hybrid Recommendations ---
        print("\n--- Testing Hybrid Recommendations ---")
        # Example: Get recommendations for a specific user (e.g., User ID 1)
        sample_user_id = 1 
        
        # Check if user_id exists in our users_df
        if sample_user_id not in users_df_raw['user_id'].values:
            print(f"User ID {sample_user_id} not found in users_df_raw. Please choose an existing user ID.")
            # If user does not exist, find an existing one to test
            sample_user_id = users_df_raw['user_id'].iloc[0]
            print(f"Using User ID {sample_user_id} for demonstration.")
            
        recommended_places = get_hybrid_recommendations(
            user_id=sample_user_id,
            cf_model=trained_cf_model,
            places_df=places_df_raw,
            content_features_df=full_content_features_df,
            tfidf_vectorizer=tfidf_vectorizer,
            ohe_encoder=ohe_encoder,
            ratings_df=ratings_df_raw, # <--- NEW: Pass ratings_df_raw here
            num_recommendations=5
        )

        if not recommended_places.empty:
            print(f"\nTop {len(recommended_places)} Hybrid Recommendations for User ID {sample_user_id}:")
            print(recommended_places[['place_id', 'place_name', 'category', 'city', 'rating']])
        else:
            print(f"Could not generate recommendations for User ID {sample_user_id}.")

        # Example: Test with a cold-start user ID (assuming this ID does not exist in ratings_df)
        cold_start_user_id = 9999
        print(f"\nTesting hybrid recommendations for a cold-start user (ID: {cold_start_user_id})...")
        cold_start_recs = get_hybrid_recommendations(
            user_id=cold_start_user_id,
            cf_model=trained_cf_model,
            places_df=places_df_raw,
            content_features_df=full_content_features_df,
            tfidf_vectorizer=tfidf_vectorizer,
            ohe_encoder=ohe_encoder,
            ratings_df=ratings_df_raw, # <--- NEW: Pass ratings_df_raw here
            num_recommendations=5
        )
        if not cold_start_recs.empty:
            print(f"\nTop {len(cold_start_recs)} Hybrid Recommendations for Cold-Start User ID {cold_start_user_id}:")
            print(cold_start_recs[['place_id', 'place_name', 'category', 'city', 'rating']])
        else:
            print(f"Could not generate recommendations for Cold-Start User ID {cold_start_user_id}.")
        
        print("\nHybrid recommendation testing complete.")