# ingest-data.py

import pandas as pd
import psycopg2
from io import StringIO
import os
import csv
import numpy as np
from psycopg2.extras import execute_values
from psycopg2.extensions import register_adapter, AsIs
register_adapter(np.int64, AsIs)

# Database connection details
DB_HOST = os.getenv('DB_HOST', 'host.docker.internal')
DB_NAME = os.getenv('DB_NAME', 'postgres')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
DB_PORT = os.getenv('DB_PORT', '5432')

# Paths to your CSV files
TOURISM_DATA_CSV = 'ml/data/tourism_with_id.csv'
RATING_DATA_CSV = 'ml/data/tourism_rating.csv'
USER_DATA_CSV = 'ml/data/user.csv'

def clean_tourism_data(df):
    """Cleans the tourism_with_id DataFrame."""
    df_cleaned = df.drop(columns=['Unnamed: 11', 'Unnamed: 12'], errors='ignore')
    df_cleaned['Description'] = df_cleaned['Description'].astype(str).str.replace(r'[\r\n]+', ' ', regex=True).str.strip()
    df_cleaned['Place_Name'] = df_cleaned['Place_Name'].astype(str).str.replace(r'[\r\n]+', ' ', regex=True).str.strip()
    df_cleaned['Time_Minutes'] = df_cleaned.groupby('Category')['Time_Minutes'].transform(lambda x: x.fillna(x.median()))
    df_cleaned['Time_Minutes'] = df_cleaned['Time_Minutes'].fillna(df_cleaned['Time_Minutes'].median())
    df_cleaned['Time_Minutes'] = df_cleaned['Time_Minutes'].astype(int)
    return df_cleaned

def insert_data_to_db(conn, df, table_name, columns):
    """Inserts DataFrame data into a PostgreSQL table using execute_values."""
    cur = conn.cursor()
    
    data_values = [tuple(row) for row in df[list(columns)].values] # Ensure df columns exist here
    sql_insert = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES %s"

    try:
        execute_values(cur, sql_insert, data_values)
        conn.commit()
        print(f"Successfully inserted {len(df)} rows into {table_name} using execute_values.")
    except (Exception, psycopg2.Error) as error:
        print(f"Error inserting data into {table_name} using execute_values: {error}")
        conn.rollback()
    finally:
        cur.close()

def truncate_table(conn, table_name):
    """Truncates a table, clearing all its data."""
    cur = conn.cursor()
    try:
        cur.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE;")
        conn.commit()
        print(f"Table '{table_name}' truncated successfully.")
    except (Exception, psycopg2.Error) as error:
        print(f"Error truncating table '{table_name}': {error}")
        conn.rollback()
    finally:
        cur.close()


def main():
    conn = None
    try:
        print(f"Connecting to database at {DB_HOST}:{DB_PORT}/{DB_NAME} as {DB_USER}...")
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("Database connection successful.")

        truncate_table(conn, 'ratings')
        truncate_table(conn, 'users')
        truncate_table(conn, 'places')
        print("All tables truncated for a fresh start.")

        print(f"Loading {TOURISM_DATA_CSV}...")
        df_tourism = pd.read_csv(TOURISM_DATA_CSV)
        df_tourism_cleaned = clean_tourism_data(df_tourism)
        print("Tourism data cleaned.")

        print(f"Loading {USER_DATA_CSV}...")
        df_user = pd.read_csv(USER_DATA_CSV)
        
        print(f"Loading {RATING_DATA_CSV}...")
        df_rating = pd.read_csv(RATING_DATA_CSV)

        initial_ratings_count = len(df_rating)
        df_rating.drop_duplicates(subset=['User_Id', 'Place_Id'], inplace=True)
        if len(df_rating) < initial_ratings_count:
            print(f"Removed {initial_ratings_count - len(df_rating)} duplicate ratings (User_Id, Place_Id) combinations.")

        # --- FIX: Rename DataFrame columns to match DB schema before insertion ---

        # Rename df_user columns
        df_user.rename(columns={
            'User_Id': 'user_id',
            'Location': 'location',
            'Age': 'age'
        }, inplace=True)

        # Rename df_tourism_cleaned columns
        df_tourism_cleaned.rename(columns={
            'Place_Id': 'place_id',
            'Place_Name': 'place_name',
            'Description': 'description',
            'Category': 'category',
            'City': 'city',
            'Price': 'price',
            'Rating': 'rating',
            'Time_Minutes': 'time_minutes',
            'Coordinate': 'coordinate',
            'Lat': 'lat',
            'Long': 'long'
        }, inplace=True)
        
        # df_rating already has 'Place_Ratings' renamed to 'place_rating' from previous step
        # Make sure 'User_Id' and 'Place_Id' are also renamed for consistency
        df_rating.rename(columns={
            'User_Id': 'user_id',
            'Place_Id': 'place_id',
            'Place_Ratings': 'place_rating' # This was already done, but keeping here for completeness
        }, inplace=True)

        # --- Convert numeric columns in df_rating to standard Python int ---
        print("Converting rating columns to standard Python integers...")
        df_rating['user_id'] = df_rating['user_id'].astype(int)
        df_rating['place_id'] = df_rating['place_id'].astype(int)
        df_rating['place_rating'] = df_rating['place_rating'].astype(int)
        print("Rating columns converted.")
        # --- END ---

        # --- Data Ingestion Order Matters Due to Foreign Keys ---
        # 1. Insert users
        print("Inserting users data...")
        insert_data_to_db(conn, df_user, 'users', ('user_id', 'location', 'age'))

        # 2. Insert places
        print("Inserting places data...")
        places_db_columns = ('place_id', 'place_name', 'description', 'category', 'city', 'price', 'rating', 'time_minutes', 'coordinate', 'lat', 'long')
        insert_data_to_db(conn, df_tourism_cleaned, 'places', places_db_columns)

        # 3. Insert ratings
        print("Inserting ratings data...")
        ratings_db_columns = ('user_id', 'place_id', 'place_rating')
        insert_data_to_db(conn, df_rating, 'ratings', ratings_db_columns)


    except (Exception, psycopg2.Error) as error:
        print(f"An error occurred during data ingestion: {error}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()