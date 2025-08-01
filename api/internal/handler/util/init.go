package util

import (
	"github.com/hanifbg/IndonesiaTourismDestination/config"
	"github.com/hanifbg/IndonesiaTourismDestination/internal/handler/place"
	serv "github.com/hanifbg/IndonesiaTourismDestination/internal/service/util"
	"github.com/labstack/echo"
)

func InitHandler(config *config.AppConfig, e *echo.Echo, servWrapper *serv.ServiceWrapper) {
	place.InitRoute(e, servWrapper)
}
