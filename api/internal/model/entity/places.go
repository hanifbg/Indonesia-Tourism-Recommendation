package entity

import "time"

// Place represents a tourism destination from the 'places' table
type Place struct {
	PlaceID     int     `json:"place_id"`
	PlaceName   string  `json:"place_name"`
	Description string  `json:"description"`
	Category    string  `json:"category"`
	City        string  `json:"city"`
	Price       float64 `json:"price"`
	Rating      float64 `json:"rating"`
	TimeMinutes float64 `json:"time_minutes"`
	Coordinate  string  `json:"coordinate"` // Stored as string, as Lat/Long are separate
	Lat         float64 `json:"lat"`
	Long        float64 `json:"long"`
}

// Rating represents a user's rating for a place from the 'ratings' table
type Rating struct {
	UserID          int       `json:"user_id"`
	PlaceID         int       `json:"place_id"`
	PlaceRating     int       `json:"place_rating"`
	RatingTimestamp time.Time `json:"rating_timestamp"`
}

// RecommendationRequestML is the struct for the request body sent to the Python ML Inference Service
type RecommendationRequestML struct {
	UserID             int `json:"user_id"`
	NumRecommendations int `json:"num_recommendations"`
}

// RecommendationResponseML is the struct for the response body from the Python ML Inference Service
type RecommendationResponseML []Place // The ML service returns a list of Place objects

// PlaceSearchQueryParams represents query parameters for place search
type PlaceSearchQueryParams struct {
	Query string `query:"query"`
}

// RecommendationAPIRequest represents the request for the main Go API's recommendation endpoint
type RecommendationAPIRequest struct {
	UserID int `json:"user_id"`
	Limit  int `json:"limit"` // To control number of recommendations
}
