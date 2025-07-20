# explore_data.py
import pandas as pd
import psycopg2
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np 

# Database connection details (same as ingest-data.py)
DB_HOST = os.getenv('DB_HOST', 'host.docker.internal')
DB_NAME = os.getenv('DB_NAME', 'postgres') # Ensure this matches your actual DB name
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'password') # Use your actual password
DB_PORT = os.getenv('DB_PORT', '5432') # Ensure this matches your actual exposed port for PostgreSQL

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
    """Loads data from PostgreSQL tables into Pandas DataFrames."""
    if conn is None:
        print("No database connection available.")
        return None, None, None

    df_places = None
    df_users = None
    df_ratings = None

    try:
        print("Loading data from 'places' table...")
        df_places = pd.read_sql("SELECT * FROM places;", conn)
        print(f"Loaded {len(df_places)} rows from 'places'.")

        print("Loading data from 'users' table...")
        df_users = pd.read_sql("SELECT * FROM users;", conn)
        print(f"Loaded {len(df_users)} rows from 'users'.")

        print("Loading data from 'ratings' table...")
        df_ratings = pd.read_sql("SELECT * FROM ratings;", conn)
        print(f"Loaded {len(df_ratings)} rows from 'ratings'.")

    except Exception as e:
        print(f"Error loading data from database: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")
    
    return df_places, df_users, df_ratings

if __name__ == "__main__":
    conn = get_db_connection()
    if conn:
        places_df, users_df, ratings_df = load_data_from_db(conn)

        if places_df is not None:
            print("\n--- Places DataFrame Info ---")
            print(places_df.info())
            print("\n--- Places Head ---")
            print(places_df.head())

            # --- NEW EDA ADDITIONS START HERE for Places ---
            print("\n--- Places: Top Categories ---")
            print(places_df['category'].value_counts().head(10)) # Top 10 categories

            print("\n--- Places: Top Cities ---")
            print(places_df['city'].value_counts().head(10)) # Top 10 cities

            print("\n--- Places: Price Distribution ---")
            print(places_df['price'].describe())

            print("\n--- Places: Rating Distribution ---")
            print(places_df['rating'].describe())

            print("\n--- Places: Time_Minutes Distribution ---")
            print(places_df['time_minutes'].describe())
            # --- NEW EDA ADDITIONS END HERE for Places ---
        
        if users_df is not None:
            print("\n--- Users DataFrame Info ---")
            print(users_df.info())
            print("\n--- Users Head ---")
            print(users_df.head())

            # --- NEW EDA ADDITIONS START HERE for Users ---
            print("\n--- Users: Top Locations ---")
            print(users_df['location'].value_counts().head(10)) # Top 10 locations

            print("\n--- Users: Age Distribution ---")
            print(users_df['age'].describe())
            # --- NEW EDA ADDITIONS END HERE for Users ---

        if ratings_df is not None:
            print("\n--- Ratings DataFrame Info ---")
            print(ratings_df.info())
            print("\n--- Ratings Head ---")
            print(ratings_df.head())

            # --- NEW EDA ADDITIONS START HERE for Ratings ---
            print("\n--- Ratings: Place_Rating Distribution ---")
            print(ratings_df['place_rating'].describe())
            print(ratings_df['place_rating'].value_counts().sort_index()) # Count of each rating (1-5)

            print("\n--- Ratings: Number of ratings per Place (Top 10 Most Rated) ---")
            print(ratings_df['place_id'].value_counts().head(10))

            print("\n--- Ratings: Number of ratings per User (Top 10 Most Active) ---")
            print(ratings_df['user_id'].value_counts().head(10))
            
            # --- Deeper Merged Data Analysis ---
            # Calculate average rating per place from user ratings
            print("\n--- Merged Data: Average Place Rating by Category (Top 10) ---")
            avg_rating_per_place = ratings_df.groupby('place_id')['place_rating'].mean().reset_index()
            avg_rating_per_place.rename(columns={'place_rating': 'avg_user_rating'}, inplace=True)
            
            # Merge with places_df to get category information
            places_with_avg_user_rating = pd.merge(places_df, avg_rating_per_place, on='place_id', how='left')
            
            # Now, average these 'avg_user_rating' by category
            # We use .dropna() because some places might not have user ratings yet (left merge)
            avg_rating_by_category = places_with_avg_user_rating.groupby('category')['avg_user_rating'].mean().sort_values(ascending=False).head(10)
            print(avg_rating_by_category)
            
            print("\n--- Merged Data: Average Place Rating by City (Top 10) ---")
            avg_rating_by_city = places_with_avg_user_rating.groupby('city')['avg_user_rating'].mean().sort_values(ascending=False).head(10)
            print(avg_rating_by_city)
            # --- NEW EDA ADDITIONS END HERE for Ratings ---

        # --- Phase 3.2: Feature Engineering Starts Here ---
        print("\n\n--- Starting Feature Engineering ---")

        if places_df is not None:
            # 1. Text Vectorization (TF-IDF) for 'Description' and 'Place_Name'
            # We will combine Description and Place_Name for richer text features
            places_df['combined_text'] = places_df['place_name'] + " " + places_df['description']
            
            print("\n--- Text Feature Engineering (TF-IDF) ---")
            tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=3) # Ignore words appearing less than 3 times
            tfidf_matrix = tfidf_vectorizer.fit_transform(places_df['combined_text'])
            
            print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
            # print("Sample TF-IDF Features (first 5 features, first 5 places):")
            # print(tfidf_matrix[:5, :5].toarray()) # Convert sparse matrix to dense array for printing

            # 2. Categorical Encoding (One-Hot Encoding) for 'category' and 'city'
            print("\n--- Categorical Feature Engineering (One-Hot Encoding) ---")
            # Create a OneHotEncoder
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False for dense array
            
            # Fit and transform 'category' and 'city'
            # First, select the columns to encode
            categorical_cols = ['category', 'city']
            encoded_features = ohe.fit_transform(places_df[categorical_cols])
            
            # Create a DataFrame from encoded features with meaningful column names
            encoded_feature_names = ohe.get_feature_names_out(categorical_cols)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=places_df.index)
            
            print(f"One-Hot Encoded Features Shape: {encoded_df.shape}")
            # print("Sample One-Hot Encoded Features (first 5 places):")
            # print(encoded_df.head())

            # 3. Numerical Features (from places_df)
            print("\n--- Numerical Features (from Places DataFrame) ---")
            numerical_cols = ['price', 'rating', 'time_minutes', 'lat', 'long']
            numerical_df = places_df[numerical_cols].copy() # Ensure we're working on a copy
            
            print(f"Numerical Features Shape: {numerical_df.shape}")
            # print("Sample Numerical Features (first 5 places):")
            # print(numerical_df.head())


            # 4. Combining all Content-Based Features for 'places_df'
            print("\n--- Combining Content-Based Features ---")
            # Convert TF-IDF sparse matrix to DataFrame for concatenation
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=places_df.index)
            
            # Concatenate all features horizontally
            # Ensure all DataFrames have the same index before concatenating
            content_features_df = pd.concat([tfidf_df, encoded_df, numerical_df], axis=1)
            
            print(f"Combined Content-Based Features Shape: {content_features_df.shape}")
            print("Combined Content-Based Features Sample (first 5 places, first 5 columns + last 5 columns):")
            # Print a small slice to avoid overwhelming output, focusing on start and end of features
            # Adjust column selection to avoid printing too much if there are many TF-IDF features
            if content_features_df.shape[1] > 10:
                 print(content_features_df.iloc[:5, list(range(5)) + list(range(content_features_df.shape[1]-5, content_features_df.shape[1]))])
            else:
                 print(content_features_df.head())

            print("\n--- Content-Based Feature Engineering Completed ---")

        # --- Collaborative Filtering Data Preparation (from ratings_df) ---
        if ratings_df is not None:
            print("\n--- Collaborative Filtering Data Preparation ---")
            # For CF, the primary data is the user-item-rating triplets.
            # Libraries like Surprise expect a specific format: (user_id, item_id, rating)
            # Our ratings_df already has 'user_id', 'place_id', 'place_rating'
            
            print("Ratings DataFrame is already in (user_id, place_id, place_rating) format for CF.")
            print("Sample Ratings Data for CF (first 5 rows):")
            print(ratings_df[['user_id', 'place_id', 'place_rating']].head())
            print(f"Number of unique users: {ratings_df['user_id'].nunique()}")
            print(f"Number of unique places rated: {ratings_df['place_id'].nunique()}")
            print("\n--- Collaborative Filtering Data Preparation Completed ---")

        print("\n\n--- Feature Engineering All Components Completed ---")