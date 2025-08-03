-- db/init.sql

-- Create the tables as per our defined schema
CREATE TABLE places (
    place_id INTEGER PRIMARY KEY,
    place_name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    price INTEGER NOT NULL,
    rating NUMERIC(2,1) NOT NULL,
    time_minutes INTEGER NOT NULL,
    coordinate VARCHAR(255) NOT NULL,
    lat NUMERIC(10,7) NOT NULL,
    long NUMERIC(10,7) NOT NULL
);

CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    location VARCHAR(255) NOT NULL,
    age INTEGER NOT NULL
);

CREATE TABLE ratings (
    user_id INTEGER NOT NULL,
    place_id INTEGER NOT NULL,
    place_rating INTEGER NOT NULL,
    rating_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, place_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (place_id) REFERENCES places(place_id),
    CHECK (place_rating BETWEEN 1 AND 5)
);

-- Note: The `postgres` image's entrypoint automatically creates the DB from the `POSTGRES_DB` env var,
-- so we just need to create the tables in this script.