package place

import (
	"net/http"
	"strconv"

	"github.com/labstack/echo"
)

func (a *ApiWrapper) GetPlaceById(c echo.Context) error {
	ids := c.Param("id")
	id, err := strconv.Atoi(ids)
	if err != nil {
		return err
	}

	place, err := a.PlaceService.GetPlaceByID(id)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"message": "Internal server error",
		})
	}

	return c.JSON(http.StatusOK, place)
}
