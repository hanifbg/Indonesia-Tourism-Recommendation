package entity

// User represents a user from the 'users' table
type User struct {
	UserID   int    `json:"user_id"`
	Location string `json:"location"`
	Age      int    `json:"age"`
}
