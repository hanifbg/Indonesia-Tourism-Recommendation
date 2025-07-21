# Capstone Project: Indonesian Tourism Destination Recommender System

## 1. Project Overview

This capstone project focuses on developing a robust and production-ready hybrid machine learning recommendation system for Indonesian tourism destinations. The system is designed to provide personalized suggestions by combining content-based filtering (recommending similar places) and collaborative filtering (recommending based on user preferences).

The development process adheres to MLOps best practices, emphasizing reproducibility, automation, and deployability. The final system will be deployed using cloud services.

**Key Objectives:**
1.  **Real-World Application:** Develop a functional recommendation system for practical use.
2.  **Independent Problem Solving:** Demonstrate the ability to tackle complex technical challenges inherent in ML system development.
3.  **Professional Portfolio Development:** Create a demonstrable project showcasing expertise in machine learning and MLOps.

## 2. Current Progress: Data Foundation & Core ML Model Development

This project has successfully established its data foundation, completed comprehensive data analysis, engineered features, and developed and trained the core hybrid machine learning model, integrated with MLflow.

### 2.1. Database Setup & Data Ingestion

**Objective:** To establish a persistent and structured storage for the tourism data and to load the initial datasets reliably.

**Implementation Details:**
1.  **Local PostgreSQL Setup:** A PostgreSQL database has been set up locally using Docker (specifically, the `deploycamp` container exposed on port `5432`). This approach ensures an isolated and consistent database environment, which is a key MLOps best practice for development.
2.  **Database Schema Definition:** A robust relational schema comprising three tables (`places`, `users`, `ratings`) has been designed and implemented. Primary and foreign keys are utilized to enforce data integrity and define relationships between the entities.
3.  **Dockerized Data Ingestion Script (`ml/scripts/ingest_data.py`):**
    * A Python script has been developed to automate the entire data ingestion process.
    * **Process Overview:**
        * The script connects to the PostgreSQL database using `psycopg2`.
        * **Table Truncation:** For development purposes, all data in the `ratings`, `users`, and `places` tables is truncated at the start of each run. This ensures a clean and reproducible state, preventing issues like duplicate key errors from prior runs.
        * **CSV Data Loading:** Raw data from `tourism_with_id.csv`, `tourism_rating.csv`, and `user.csv` is loaded into Pandas DataFrames.
        * **Data Cleaning (Tourism Data):**
            * Irrelevant `Unnamed` columns from `tourism_with_id.csv` are removed.
            * Missing values in the `Time_Minutes` column are imputed using the median duration specific to each `Category`. A fallback to the overall median is used if a category has no valid data.
            * Newline and carriage return characters (`\n`, `\r`) within `Description` and `Place_Name` text fields are replaced with spaces. This prevents issues during database insertion.
        * **Rating Deduplication:** Duplicate `(user_id, place_id)` pairs are identified and removed from the `ratings` DataFrame (`df.drop_duplicates()`), ensuring that each user has only one unique rating per place, adhering to the database's primary key constraint.
        * **Column Renaming:** DataFrame column names (e.g., `User_Id`) are systematically renamed to precisely match the PostgreSQL table column names (e.g., `user_id`, using lowercase and underscores).
        * **Robust Insertion Method:** Data from Pandas DataFrames is inserted into PostgreSQL tables using `psycopg2.extras.execute_values`. This method is highly efficient for bulk inserts and automatically handles SQL parameter escaping and type conversions, preventing various quoting and type adaptation errors.
    * **Dockerization:** The `ml/Dockerfile.ingest` packages this ingestion script along with its Python dependencies (`pandas`, `psycopg2-binary`) into a Docker image (`data-ingester`). This approach guarantees the ingestion process is reproducible and consistent across different environments.

### 2.2. Exploratory Data Analysis (EDA) & Feature Engineering

**Objective:** To gain deep insights into the characteristics of the data and to transform raw attributes into numerical features suitable for machine learning models.

**Implementation Details:**
1.  **Dockerized EDA Script (`ml/scripts/explore_data.py`):**
    * A Python script has been developed to connect to the PostgreSQL database and load the `places`, `users`, and `ratings` data into Pandas DataFrames for analysis.
    * **Process Overview:**
        * **Basic Data Overview:** The script outputs `df.info()` and `df.head()` for all loaded DataFrames, confirming data structure and successful retrieval.
        * **Distribution Analysis:** Summary statistics (`describe()`) and frequency counts (`value_counts()`) are generated for key numerical and categorical columns. This analysis provided insights into:
            * The prevalence of different place categories and cities.
            * Price ranges and rating distributions.
            * User demographics and activity levels.
        * **Relationship Insights:** Initial analyses included calculating the number of ratings per place and per user, as well as average user ratings aggregated by category and city.
    * **Dockerization:** The `ml/Dockerfile.ml` packages this script (and other ML scripts) with all required Python dependencies (`pandas`, `psycopg2-binary`, `scikit-learn`, `scikit-surprise`, `mlflow`) into an `ml-env` Docker image, ensuring a consistent execution environment for all ML-related Python code.

2.  **Feature Engineering Pipelines:** The `explore_data.py` script also performs the initial feature engineering transformations, preparing the data for the hybrid recommender system:
    * **Content-Based Features (from `places` data):**
        * **Text Vectorization (TF-IDF):** `place_name` and `description` fields are combined and processed using `TfidfVectorizer` to create numerical feature vectors.
        * **Categorical Encoding (One-Hot Encoding):** `category` and `city` columns are transformed using `OneHotEncoder` into binary features.
        * **Numerical Features:** Existing numerical features (`price`, `rating`, `time_minutes`, `lat`, `long`) are directly included.
        * **Combined Content Features:** All content-based features are concatenated into a single `content_features_df`. This composite matrix serves as the rich content representation for our content-based recommendation component.
    * **Collaborative Filtering Data Preparation (from `ratings` data):**
        * The `ratings` DataFrame is confirmed to be in the `(user_id, place_id, place_rating)` triplet format, which is the standard input for collaborative filtering models.

### 2.3. Core ML Model Development & MLflow Tracking

**Objective:** To implement, train, and evaluate the hybrid recommendation model, and to integrate robust experiment tracking using MLflow.

**Implementation Details:**
1.  **Dockerized Model Training Script (`ml/scripts/train_model.py`):**
    * This script orchestrates the entire model development process from data loading to model saving.
    * **Process Overview:**
        * Loads cleaned data from the PostgreSQL database.
        * Performs feature engineering (as detailed in 2.2).
        * **Data Splitting:** Splits the ratings data into training and test sets (e.g., 80% train, 20% test) using `surprise.model_selection.train_test_split` for collaborative filtering.
        * **Collaborative Filtering (CF) Model Training:** Trains a Singular Value Decomposition (SVD) model from the `scikit-surprise` library on the training set.
        * **CF Model Evaluation:** Evaluates the trained SVD model on the test set using standard metrics like RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
        * **Hybrid Recommendation Logic:** Implements the `get_hybrid_recommendations` function, which:
            * Generates predictions from the trained CF model.
            * Calculates content-based similarities (using `cosine_similarity`) for places based on user preferences.
            * Combines CF predictions and Content-Based scores using a weighted approach to produce final hybrid recommendations.
            * Includes robust logic for handling **"cold-start" users** (users with no previous ratings) by providing popularity-based or content-based fallback recommendations.
        * **MLflow Tracking Integration:**
            * Explicitly sets an MLflow experiment named "Tourism Recommendation System."
            * Wraps the training and evaluation process within an `mlflow.start_run()` block.
            * Logs model hyperparameters (e.g., SVD type, split sizes) using `mlflow.log_param()`.
            * Logs evaluation metrics (RMSE, MAE) using `mlflow.log_metric()`.
            * **Artifact Logging:** Saves and logs the trained CF model (`cf_model.pkl`), the TF-IDF vectorizer (`tfidf_vectorizer.pkl`), and the One-Hot Encoder (`ohe_encoder.pkl`) as artifacts within the MLflow run. This ensures that the exact components used for a specific model version are easily retrievable for inference or further analysis.
    * **MLflow UI:** The MLflow Tracking server runs locally (`http://localhost:5000`), allowing real-time visualization of all experiment runs, their parameters, metrics, and stored artifacts.

## 3. How to Run the Current Setup

To replicate the current data setup and run the ML model development pipeline, ensure you have Docker Desktop installed and running.

1.  **Navigate to your project root directory** (e.g., `your-capstone-project/`).
2.  **Start your PostgreSQL Docker container:**
    ```bash
    docker run --name deploycamp -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres:17
    # Adjust password and container name if different from your setup.
    ```
3.  **Create the Database and Tables:** If you haven't yet, connect to your PostgreSQL instance and run the `CREATE DATABASE` and `CREATE TABLE` SQL commands.
4.  **Run the Data Ingestion Script:**
    * Build the `data-ingester` Docker image:
        ```bash
        docker build -t data-ingester -f ml/Dockerfile.ingest .
        ```
    * Execute the data ingestion:
        ```bash
        docker run -e DB_PASSWORD=password data-ingester # Use your actual DB password.
        ```
5.  **Start the MLflow Tracking Server:**
    * Create MLflow data directory: `mkdir -p ml/mlruns`
    * Run the MLflow UI (from your host terminal, not Docker):
        ```bash
        mlflow ui --backend-store-uri file:./ml/mlruns --host 0.0.0.0 --port 5000
        ```
    * Keep this terminal window open.
6.  **Run the Model Training and Evaluation Script:**
    * Build the `ml-env` Docker image (if not already built or `ml/requirements.txt` has changed):
        ```bash
        docker build -t ml-env -f ml/Dockerfile.ml .
        ```
    * Execute the training script, linking it to the MLflow UI:
        ```bash
        docker run \
          -e DB_PASSWORD=password \
          -e MLFLOW_TRACKING_URI=[http://host.docker.internal:5000](http://host.docker.internal:5000) \
          ml-env python ml/scripts/train_model.py # Use your actual DB password.
        ```
    * After running, check `http://localhost:5000` in your browser to see the logged experiment.

## 4. Next Steps

With the core hybrid ML model developed, trained, evaluated, and tracked, the project is now ready for **Phase 6: API Development & Integration**. The immediate next focus will be:

* **Building the Python ML Inference Service:** This will be a lightweight web service (e.g., using FastAPI) responsible for loading the trained models and feature engineering tools (TF-IDF, OHE) from MLflow artifacts, and exposing a simple API endpoint for recommendation inference.
* **Developing the Go API Service:** This main Go application (using Echo framework) will then communicate with the Python ML Inference Service to retrieve recommendations, handle primary API logic, and interact with the PostgreSQL database.

---