package place

import (
	"github.com/hanifbg/IndonesiaTourismDestination/internal/service"
	"github.com/hanifbg/IndonesiaTourismDestination/internal/service/util"
	"github.com/labstack/echo"
)

type ApiWrapper struct {
	PlaceService service.PlaceService
}

func InitRoute(e *echo.Echo, servWrapper *util.ServiceWrapper) {
	api := ApiWrapper{
		PlaceService: servWrapper.PlaceService,
	}
	api.registerRouter(e)
}

func (a *ApiWrapper) registerRouter(e *echo.Echo) {
	group := e.Group("/api/v1/place")
	// group.GET("/place", a.GetAllPlaces)
	group.GET("/:id", a.GetPlaceById)
	group.GET("/recommendations", a.GetRecommendations)
}
