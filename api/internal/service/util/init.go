package util

import (
	"github.com/hanifbg/IndonesiaTourismDestination/config"
	"github.com/hanifbg/IndonesiaTourismDestination/internal/repository/util"
	"github.com/hanifbg/IndonesiaTourismDestination/internal/service"
	"github.com/hanifbg/IndonesiaTourismDestination/internal/service/place"
)

type ServiceWrapper struct {
	PlaceService service.PlaceService
}

func New(config *config.AppConfig, repo *util.RepoWrapper) *ServiceWrapper {
	return &ServiceWrapper{
		PlaceService: place.New(config, repo),
	}
}
