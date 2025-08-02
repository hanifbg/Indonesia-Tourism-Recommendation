package mlapi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/hanifbg/IndonesiaTourismDestination/internal/model/entity"
)

func (c *MLApiClient) GetRecommendations(request *entity.RecommendationRequestML) (entity.RecommendationResponseML, error) {
	requestURL := fmt.Sprintf("http://localhost:8000/recommend")

	jsonBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	// Create and send the HTTP request
	resp, err := c.Client.Post(requestURL, "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to send request to ML service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		log.Printf("ML service returned non-200 status: %d, body: %s", resp.StatusCode, string(bodyBytes))
		return nil, fmt.Errorf("ML service returned status: %d", resp.StatusCode)
	}

	// Decode the response body
	var recommendations []entity.Place
	if err := json.NewDecoder(resp.Body).Decode(&recommendations); err != nil {
		return nil, fmt.Errorf("failed to decode response from ML service: %w", err)
	}

	return recommendations, nil
}
