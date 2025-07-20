# Capstone Project: Indonesian Tourism Destination Recommender System

## 1. Project Overview

This capstone project focuses on developing a robust and production-ready hybrid machine learning recommendation system for Indonesian tourism destinations. The system is designed to provide personalized suggestions by combining content-based filtering (recommending similar places) and collaborative filtering (recommending based on user preferences).

The development process adheres to MLOps best practices, emphasizing reproducibility, automation, and deployability. The final system will be deployed using cloud services.

**Key Objectives:**
1.  **Real-World Application:** Develop a functional recommendation system for practical use.
2.  **Independent Problem Solving:** Demonstrate the ability to tackle complex technical challenges inherent in ML system development.
3.  **Professional Portfolio Development:** Create a demonstrable project showcasing expertise in machine learning and MLOps.

## 2. Current Progress: Data Foundation & Feature Engineering

This project has successfully completed the foundational phases of data management and is well into preparing data for machine learning models.

### 2.1. Database Setup & Data Ingestion

**Objective:** To establish a persistent and structured storage for the tourism data and to load the initial datasets reliably.

**Implementation Details:**
1.  **Local PostgreSQL Setup:** A PostgreSQL database has been set up locally using Docker (specifically, the `postgres` container exposed on port `5432`). This approach ensures an isolated and consistent database environment, which is a key MLOps best practice for development.
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
            * Newline and carriage return characters (`\n`, `\r`) within `Description` and `Place_Name` text fields are replaced with spaces. This prevents issues during database insertion (e.g., "literal carriage return found in data" errors).
        * **Rating Deduplication:** Duplicate `(user_id, place_id)` pairs are identified and removed from the `ratings` DataFrame. This ensures adherence to the database's unique constraint, allowing each user only one unique rating per place.
        * **Column Renaming:** DataFrame column names (e.g., `User_Id`) are systematically renamed to precisely match the PostgreSQL table column names (e.g., `user_id`, using lowercase and underscores). This ensures seamless data mapping during insertion.
        * **Robust Insertion Method:** Data from Pandas DataFrames is inserted into PostgreSQL tables using `psycopg2.extras.execute_values`. This method is highly efficient for bulk inserts and automatically handles SQL parameter escaping and type conversions (e.g., converting NumPy integers to native PostgreSQL integers), thereby preventing various quoting and type adaptation errors.
    * **Dockerization:** The `ml/Dockerfile.ingest` packages this ingestion script along with its Python dependencies (`pandas`, `psycopg2-binary`) into a Docker image (`data-ingester`). This approach guarantees the ingestion process is reproducible and consistent across different environments.

### 2.2. Exploratory Data Analysis (EDA) & Feature Engineering

**Objective:** To gain deep insights into the characteristics of the data and to transform raw attributes into numerical features suitable for machine learning models.

**Implementation Details:**
1.  **Dockerized EDA Script (`ml/scripts/explore_data.py`):**
    * A Python script has been developed to connect to the PostgreSQL database and load the `places`, `users`, and `ratings` data into Pandas DataFrames for analysis.
    * **Process Overview:**
        * **Basic Data Overview:** The script outputs `df.info()` and `df.head()` for all loaded DataFrames, confirming data structure and successful retrieval.
        * **Distribution Analysis:** Summary statistics (`describe()`) and frequency counts (`value_counts()`) are generated for key numerical (`price`, `rating`, `time_minutes`, `age`) and categorical (`category`, `city`, `location`, `place_rating`) columns. This analysis revealed:
            * The prevalence of `Taman Hiburan`, `Budaya`, and `Cagar Alam` categories.
            * The main cities covered: Yogyakarta, Bandung, Jakarta, Semarang, and Surabaya.
            * The skewed distribution of prices (many free places, high max price).
            * The generally high inherent ratings of places.
            * The age distribution of users (primarily young adults).
            * The relatively even distribution of user ratings (1-5 scale), indicating meaningful user feedback.
        * **Relationship Insights:** Initial analyses included calculating the number of ratings per place and per user, as well as average user ratings aggregated by category and city. This provided early insights into popular attractions, active users, and general preferences.
    * **Dockerization:** The `ml/Dockerfile.ml` packages this script (and will be used for future ML scripts) with all required Python dependencies (`pandas`, `psycopg2-binary`, `scikit-learn`, `scikit-surprise`) into a `ml-env` Docker image, ensuring a consistent execution environment for all ML-related Python code.

2.  **Feature Engineering Pipelines:** The `explore_data.py` script also performs the initial feature engineering transformations, preparing the data for the hybrid recommender system:
    * **Content-Based Features (from `places` data):**
        * **Text Vectorization (TF-IDF):** The `place_name` and `description` fields are combined and processed using `TfidfVectorizer`. This converts the text into a sparse numerical matrix (e.g., `437 places` x `1787 TF-IDF features`), where each feature represents the importance of a word. This allows the model to understand semantic similarities between places based on their textual content.
        * **Categorical Encoding (One-Hot Encoding):** The `category` and `city` columns are transformed using `OneHotEncoder`, resulting in new binary features (e.g., `11` features). This numerically represents categorical attributes, enabling distance-based calculations.
        * **Numerical Features:** Existing numerical features (`price`, `rating`, `time_minutes`, `lat`, `long`) are directly included.
        * **Combined Content Features:** All content-based features (TF-IDF vectors, One-Hot encoded categories/cities, and numerical attributes) are concatenated into a single `content_features_df` (e.g., `437 places` x `1803 features`). This composite matrix serves as the rich content representation for our content-based recommendation component.
    * **Collaborative Filtering Data Preparation (from `ratings` data):**
        * The `ratings` DataFrame is confirmed to be in the `(user_id, place_id, place_rating)` triplet format. This is the standard input format required by collaborative filtering libraries for building user-item interaction matrices. The script verifies the counts of unique users (300) and unique places (437) that have interactions, providing the dimensions for the user-item matrix that the CF model will learn from.

## 3. How to Run the Current Setup

To replicate the current data setup and run the EDA/Feature Engineering script, ensure you have Docker Desktop installed and running.

1.  **Navigate to your project root directory** (e.g., `your-capstone-project/`).
2.  **Start your PostgreSQL Docker container:**
    ```bash
    docker run --name postgres -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres:17
    # Adjust password and container name if different from your setup.
    ```
3.  **Create the Database and Tables:** Connect to your PostgreSQL instance (e.g., using `docker exec -it postgres psql -U postgres -d deploycamp`) and run the `CREATE DATABASE` and `CREATE TABLE` SQL commands as defined in the project's documentation (schema is in previous discussion logs).
4.  **Run the Data Ingestion Script:**
    * Build the `data-ingester` Docker image:
        ```bash
        docker build -t data-ingester -f ml/Dockerfile.ingest .
        ```
    * Execute the data ingestion:
        ```bash
        docker run -e DB_PASSWORD=password data-ingester # Use your actual DB password.
        ```
5.  **Run the EDA and Feature Engineering Script:**
    * Build the `ml-env` Docker image (if not already built or `ml/requirements.txt` has changed):
        ```bash
        docker build -t ml-env -f ml/Dockerfile.ml .
        ```
    * Execute the EDA and Feature Engineering script:
        ```bash
        docker run -e DB_PASSWORD=password ml-env python ml/scripts/explore_data.py # Use your actual DB password.
        ```
    * The console output will display the detailed EDA insights and the shapes of the engineered features.

## 4. Next Steps

With the data foundation solid and features engineered, the project is now ready for **Phase 4: Model Development & Training (Hybrid Recommender)**. This upcoming phase will involve:

* Splitting the data into appropriate training, validation, and test sets.
* Implementing and training the Collaborative Filtering model.
* Defining the Content-Based similarity logic.
* Developing the strategy for combining the two recommendation approaches into a hybrid system.
* Integrating **MLflow** for robust experiment tracking and model versioning, adhering to MLOps best practices.