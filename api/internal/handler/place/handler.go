package place

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/hanifbg/IndonesiaTourismDestination/internal/model/entity"
	"github.com/labstack/echo"
)

func (a *ApiWrapper) GetPlaceById(c echo.Context) error {
	ids := c.Param("id")
	id, err := strconv.Atoi(ids)
	if err != nil {
		return err
	}

	place, err := a.PlaceService.GetPlaceByID(id)
	fmt.Println(err)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"message": "Internal server error",
		})
	}

	return c.JSON(http.StatusOK, place)
}

func (a *ApiWrapper) GetRecommendations(c echo.Context) error {
	var request entity.RecommendationRequestML

	// Parse and validate the user ID from the request
	userIDStr := c.QueryParam("user_id")
	if userIDStr == "" {
		return c.JSON(http.StatusBadRequest, map[string]string{"error": "user_id query parameter is required"})
	}
	userID, err := strconv.Atoi(userIDStr)
	if err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{"error": "Invalid user_id"})
	}
	// Parse and validate the number of recommendations
	limitStr := c.QueryParam("limit")
	limit := 10 // Default limit
	if limitStr != "" {
		limit, err = strconv.Atoi(limitStr)
		if err != nil {
			return c.JSON(http.StatusBadRequest, map[string]string{"error": "Invalid limit parameter"})
		}
	}

	request.UserID = userID
	request.NumRecommendations = limit
	response, err := a.PlaceService.GetRecommendations(&request)
	fmt.Println(err)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"message": "Internal server error",
		})
	}

	return c.JSON(http.StatusOK, response)
}
