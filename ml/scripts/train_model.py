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
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings('ignore')


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

    # --- FIX START ---
    # Ensure the ml/models directory exists BEFORE saving
    models_dir = "ml/models/"
    os.makedirs(models_dir, exist_ok=True)
    # --- FIX END ---

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

        # Log the trained CF model as a pickle artifact to MLflow and a fixed local path
        print("Logging CF model to MLflow and saving to ml/models/...")
        cf_model_local_path = os.path.join(models_dir, "cf_model.pkl") # Use os.path.join for robustness
        # os.makedirs(os.path.dirname(cf_model_local_path), exist_ok=True) # This is now handled by the early os.makedirs

        with open(cf_model_local_path, "wb") as f:
            pickle.dump(cf_model, f)
        mlflow.log_artifact(cf_model_local_path, "cf_model_artifact")

        # Log the TF-IDF Vectorizer and OneHotEncoder as artifacts to MLflow and fixed local paths
        print("Logging TF-IDF Vectorizer and OneHotEncoder to MLflow and saving to ml/models/...")
        tfidf_local_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        ohe_local_path = os.path.join(models_dir, "ohe_encoder.pkl")
        
        with open(tfidf_local_path, "wb") as f:
            pickle.dump(tfidf_vectorizer, f)
        with open(ohe_local_path, "wb") as f:
            pickle.dump(ohe_encoder, f)
        
        mlflow.log_artifact(tfidf_local_path, "feature_engineering_artifacts")
        mlflow.log_artifact(ohe_local_path, "feature_engineering_artifacts")
        print("Feature engineering tools logged to MLflow and saved to ml/models/.")

        # --- Content-Based Model Preparation ---
        print("Content-Based model: Features prepared. Similarity calculation will be done during inference.")
        
        print("\n--- Model Training & Evaluation Completed ---")
        return cf_model # Return the trained CF model

def evaluate_hybrid_system(cf_model, places_df, content_features_df, tfidf_vectorizer, ohe_encoder, ratings_df, test_users_sample=50, k_values=[5, 10, 20]):
    """
    Comprehensive evaluation of the hybrid recommendation system.
    Measures Precision@K, Recall@K, NDCG@K, and Hit Rate.
    """
    print("\n--- Starting Hybrid System Evaluation ---")
    
    # Get users who have sufficient ratings for evaluation
    user_rating_counts = ratings_df['user_id'].value_counts()
    users_with_enough_ratings = user_rating_counts[user_rating_counts >= 5].index.tolist()
    
    if len(users_with_enough_ratings) < test_users_sample:
        test_users_sample = len(users_with_enough_ratings)
        print(f"Adjusting test sample to {test_users_sample} users (all available users with >=5 ratings)")
    
    # Sample users for evaluation
    np.random.seed(42)
    test_users = np.random.choice(users_with_enough_ratings, size=test_users_sample, replace=False)
    
    evaluation_results = {k: {'precision': [], 'recall': [], 'ndcg': [], 'hit_rate': []} for k in k_values}
    
    print(f"Evaluating hybrid recommendations for {len(test_users)} users...")
    
    for user_id in test_users:
        user_ratings = ratings_df[ratings_df['user_id'] == user_id].copy()
        
        # Split user's ratings: use 80% for training context, 20% for testing
        user_ratings = user_ratings.sort_values('place_rating', ascending=False)
        n_test = max(1, len(user_ratings) // 5)  # At least 1 item for testing
        
        test_items = user_ratings.head(n_test)['place_id'].tolist()
        train_items = user_ratings.tail(len(user_ratings) - n_test)['place_id'].tolist()
        
        # Create a modified ratings_df for this evaluation (remove test items)
        eval_ratings_df = ratings_df[~((ratings_df['user_id'] == user_id) & 
                                      (ratings_df['place_id'].isin(test_items)))].copy()
        
        # Generate hybrid recommendations
        try:
            recommendations = get_hybrid_recommendations(
                user_id=user_id,
                cf_model=cf_model,
                places_df=places_df,
                content_features_df=content_features_df,
                tfidf_vectorizer=tfidf_vectorizer,
                ohe_encoder=ohe_encoder,
                ratings_df=eval_ratings_df,
                num_recommendations=max(k_values)
            )
            
            if recommendations.empty:
                continue
                
            recommended_items = recommendations['place_id'].tolist()
            
            # Calculate metrics for each K
            for k in k_values:
                top_k_recommendations = recommended_items[:k]
                
                # Precision@K
                hits = len(set(top_k_recommendations) & set(test_items))
                precision_k = hits / k if k > 0 else 0
                
                # Recall@K
                recall_k = hits / len(test_items) if len(test_items) > 0 else 0
                
                # Hit Rate@K (binary: did we hit at least one relevant item?)
                hit_rate_k = 1 if hits > 0 else 0
                
                # NDCG@K (simplified version)
                # Create relevance scores: 1 for test items, 0 for others
                relevance_scores = [1 if item in test_items else 0 for item in top_k_recommendations]
                if sum(relevance_scores) > 0 and len(relevance_scores) > 0:
                    # Ideal ranking (all relevant items first)
                    ideal_relevance = sorted(relevance_scores, reverse=True)
                    ndcg_k = ndcg_score([ideal_relevance], [relevance_scores], k=k)
                else:
                    ndcg_k = 0.0
                
                evaluation_results[k]['precision'].append(precision_k)
                evaluation_results[k]['recall'].append(recall_k)
                evaluation_results[k]['ndcg'].append(ndcg_k)
                evaluation_results[k]['hit_rate'].append(hit_rate_k)
                
        except Exception as e:
            print(f"Error evaluating user {user_id}: {e}")
            continue
    
    # Calculate average metrics
    avg_metrics = {}
    for k in k_values:
        if evaluation_results[k]['precision']:  # Check if we have any results
            avg_metrics[f'precision_at_{k}'] = np.mean(evaluation_results[k]['precision'])
            avg_metrics[f'recall_at_{k}'] = np.mean(evaluation_results[k]['recall'])
            avg_metrics[f'ndcg_at_{k}'] = np.mean(evaluation_results[k]['ndcg'])
            avg_metrics[f'hit_rate_at_{k}'] = np.mean(evaluation_results[k]['hit_rate'])
        else:
            avg_metrics[f'precision_at_{k}'] = 0.0
            avg_metrics[f'recall_at_{k}'] = 0.0
            avg_metrics[f'ndcg_at_{k}'] = 0.0
            avg_metrics[f'hit_rate_at_{k}'] = 0.0
    
    # Calculate overall hybrid system accuracy
    overall_precision = np.mean([avg_metrics[f'precision_at_{k}'] for k in k_values])
    overall_recall = np.mean([avg_metrics[f'recall_at_{k}'] for k in k_values])
    overall_ndcg = np.mean([avg_metrics[f'ndcg_at_{k}'] for k in k_values])
    overall_hit_rate = np.mean([avg_metrics[f'hit_rate_at_{k}'] for k in k_values])
    
    # F1 Score (harmonic mean of precision and recall)
    if overall_precision + overall_recall > 0:
        f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
    else:
        f1_score = 0.0
    
    avg_metrics['overall_precision'] = overall_precision
    avg_metrics['overall_recall'] = overall_recall
    avg_metrics['overall_ndcg'] = overall_ndcg
    avg_metrics['overall_hit_rate'] = overall_hit_rate
    avg_metrics['overall_f1_score'] = f1_score
    
    # Convert to percentage for easier interpretation
    hybrid_accuracy_percent = f1_score * 100
    avg_metrics['hybrid_system_accuracy_percent'] = hybrid_accuracy_percent
    
    print("\n=== HYBRID RECOMMENDATION SYSTEM EVALUATION RESULTS ===")
    print(f"Evaluated on {len(test_users)} users")
    print("\n--- Detailed Metrics by K ---")
    for k in k_values:
        print(f"\nK={k}:")
        print(f"  Precision@{k}: {avg_metrics[f'precision_at_{k}']:.4f} ({avg_metrics[f'precision_at_{k}']*100:.2f}%)")
        print(f"  Recall@{k}: {avg_metrics[f'recall_at_{k}']:.4f} ({avg_metrics[f'recall_at_{k}']*100:.2f}%)")
        print(f"  NDCG@{k}: {avg_metrics[f'ndcg_at_{k}']:.4f} ({avg_metrics[f'ndcg_at_{k}']*100:.2f}%)")
        print(f"  Hit Rate@{k}: {avg_metrics[f'hit_rate_at_{k}']:.4f} ({avg_metrics[f'hit_rate_at_{k}']*100:.2f}%)")
    
    print("\n--- Overall Hybrid System Performance ---")
    print(f"Overall Precision: {overall_precision:.4f} ({overall_precision*100:.2f}%)")
    print(f"Overall Recall: {overall_recall:.4f} ({overall_recall*100:.2f}%)")
    print(f"Overall NDCG: {overall_ndcg:.4f} ({overall_ndcg*100:.2f}%)")
    print(f"Overall Hit Rate: {overall_hit_rate:.4f} ({overall_hit_rate*100:.2f}%)")
    print(f"Overall F1-Score: {f1_score:.4f} ({f1_score*100:.2f}%)")
    print(f"\n🎯 HYBRID SYSTEM ACCURACY: {hybrid_accuracy_percent:.2f}%")
    
    print("\n--- Hybrid System Evaluation Completed ---")
    return avg_metrics

# --- Hybrid Recommendation Logic Function ---
# (No changes needed in this function for the current error, copied as is)
def get_hybrid_recommendations(
    user_id: int, 
    cf_model, 
    places_df: pd.DataFrame, 
    content_features_df: pd.DataFrame, 
    tfidf_vectorizer: TfidfVectorizer, 
    ohe_encoder: OneHotEncoder, 
    ratings_df: pd.DataFrame,
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

    user_ratings_for_cf = ratings_df[ratings_df['user_id'] == user_id]
    rated_place_ids = user_ratings_for_cf['place_id'].tolist()
    
    unrated_places = places_df[~places_df['place_id'].isin(rated_place_ids)]
    
    if len(user_ratings_for_cf) > 0:
        print(f"User {user_id} has {len(user_ratings_for_cf)} existing ratings. Using CF.")
        cf_predictions = []
        for place_id in unrated_places['place_id']:
            if cf_model.trainset.knows_item(place_id):
                predicted_rating = cf_model.predict(user_id, place_id).est
                cf_predictions.append({'place_id': place_id, 'predicted_rating': predicted_rating})
        
        cf_recs_df = pd.DataFrame(cf_predictions).sort_values(by='predicted_rating', ascending=False)
        
    else:
        print(f"User {user_id} is cold-start (no ratings). Relying on Content-Based and popularity.")
        cf_recs_df = pd.DataFrame()
        cf_weight = 0.0

    cb_recs_df = pd.DataFrame()
    
    if len(user_ratings_for_cf) > 0:
        top_user_rated_places_ids = user_ratings_for_cf.sort_values(by='place_rating', ascending=False).head(3)['place_id'].tolist()
        
        if top_user_rated_places_ids:
            place_id_to_index = {pid: idx for idx, pid in enumerate(places_df['place_id'])}
            top_rated_place_indices = [place_id_to_index[pid] for pid in top_user_rated_places_ids if pid in place_id_to_index]
            
            if top_rated_place_indices:
                top_rated_place_content_features = content_features_df.iloc[top_rated_place_indices]
                
                all_content_features_np = content_features_df.to_numpy()
                top_rated_place_content_features_np = top_rated_place_content_features.to_numpy()

                similarity_scores = cosine_similarity(top_rated_place_content_features_np, all_content_features_np)
                avg_similarity_scores = np.mean(similarity_scores, axis=0)
                
                similar_place_idx_series = pd.Series(avg_similarity_scores, index=places_df['place_id']).sort_values(ascending=False)
                
                cb_recs_series = similar_place_idx_series[~similar_place_idx_series.index.isin(rated_place_ids)].head(top_n_cb)
                
                cb_recs_df = pd.DataFrame({
                    'place_id': cb_recs_series.index,
                    'cb_score': cb_recs_series.values
                })
        else:
            print("No sufficiently top-rated places found for content-based recommendations for this user.")
            cb_recs_df = pd.DataFrame()
            
    else:
        print("Cold-start user: Recommending top-rated places overall for content-based fallback.")
        cb_recs_df = places_df.sort_values(by='rating', ascending=False).head(top_n_cb).copy()
        cb_recs_df['cb_score'] = 1.0
        cb_recs_df = cb_recs_df[~cb_recs_df['place_id'].isin(rated_place_ids)]


    final_recommendations = pd.DataFrame()
    
    if not cf_recs_df.empty and not cb_recs_df.empty:
        hybrid_df = pd.merge(cf_recs_df[['place_id', 'predicted_rating']], 
                             cb_recs_df[['place_id', 'cb_score']], 
                             on='place_id', how='outer')
        
        hybrid_df['predicted_rating'] = hybrid_df['predicted_rating'].fillna(0)
        hybrid_df['cb_score'] = hybrid_df['cb_score'].fillna(0)

        min_cf_rating, max_cf_rating = 1, 5
        hybrid_df['predicted_rating_normalized'] = (hybrid_df['predicted_rating'] - min_cf_rating) / (max_cf_rating - min_cf_rating)
        
        hybrid_df['hybrid_score'] = (cf_weight * hybrid_df['predicted_rating_normalized']) + \
                                    ((1 - cf_weight) * hybrid_df['cb_score'])
        
        final_recommendations = hybrid_df.sort_values(by='hybrid_score', ascending=False)
        
    elif not cf_recs_df.empty:
        final_recommendations = cf_recs_df.sort_values(by='predicted_rating', ascending=False)
    elif not cb_recs_df.empty:
        final_recommendations = cb_recs_df.sort_values(by='cb_score', ascending=False)
    else:
        print("No recommendations found for this user.")
        return pd.DataFrame()

    final_recommendations = final_recommendations[~final_recommendations['place_id'].isin(rated_place_ids)]
    
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

        # --- Evaluate Hybrid Recommendation System ---
        print("\n" + "="*60)
        print("COMPREHENSIVE HYBRID SYSTEM EVALUATION")
        print("="*60)
        
        # Start a new MLflow run for hybrid evaluation
        with mlflow.start_run(run_name="Hybrid_System_Evaluation"):
            hybrid_metrics = evaluate_hybrid_system(
                cf_model=trained_cf_model,
                places_df=places_df_raw,
                content_features_df=full_content_features_df,
                tfidf_vectorizer=tfidf_vectorizer,
                ohe_encoder=ohe_encoder,
                ratings_df=ratings_df_raw,
                test_users_sample=50,  # Adjust based on your dataset size
                k_values=[5, 10, 20]
            )
            
            # Log all hybrid metrics to MLflow
            for metric_name, metric_value in hybrid_metrics.items():
                mlflow.log_metric(f"hybrid_{metric_name}", round(metric_value, 4))
            
            print(f"\n✅ Hybrid evaluation metrics logged to MLflow.")

        print("\nModel training script execution complete.")
        print("Check MLflow UI at http://localhost:5000 to see experiment runs.")

        # --- Test Hybrid Recommendations ---
        print("\n--- Testing Hybrid Recommendations ---")
        sample_user_id = 1 
        
        if sample_user_id not in users_df_raw['user_id'].values:
            print(f"User ID {sample_user_id} not found in users_df_raw. Please choose an existing user ID.")
            sample_user_id = users_df_raw['user_id'].iloc[0]
            print(f"Using User ID {sample_user_id} for demonstration.")
            
        recommended_places = get_hybrid_recommendations(
            user_id=sample_user_id,
            cf_model=trained_cf_model,
            places_df=places_df_raw,
            content_features_df=full_content_features_df,
            tfidf_vectorizer=tfidf_vectorizer,
            ohe_encoder=ohe_encoder,
            ratings_df=ratings_df_raw, # Pass ratings_df_raw here
            num_recommendations=5
        )

        if not recommended_places.empty:
            print(f"\nTop {len(recommended_places)} Hybrid Recommendations for User ID {sample_user_id}:")
            print(recommended_places[['place_id', 'place_name', 'category', 'city', 'rating']])
        else:
            print(f"Could not generate recommendations for User ID {sample_user_id}.")

        cold_start_user_id = 9999
        print(f"\nTesting hybrid recommendations for a cold-start user (ID: {cold_start_user_id})...")
        cold_start_recs = get_hybrid_recommendations(
            user_id=cold_start_user_id,
            cf_model=trained_cf_model,
            places_df=places_df_raw,
            content_features_df=full_content_features_df,
            tfidf_vectorizer=tfidf_vectorizer,
            ohe_encoder=ohe_encoder,
            ratings_df=ratings_df_raw, # Pass ratings_df_raw here
            num_recommendations=5
        )
        if not cold_start_recs.empty:
            print(f"\nTop {len(cold_start_recs)} Hybrid Recommendations for Cold-Start User ID {cold_start_user_id}:")
            print(cold_start_recs[['place_id', 'place_name', 'category', 'city', 'rating']])
        else:
            print(f"Could not generate recommendations for Cold-Start User ID {cold_start_user_id}.")
        
        print("\nHybrid recommendation testing complete.")