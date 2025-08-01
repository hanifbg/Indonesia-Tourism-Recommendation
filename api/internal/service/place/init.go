package place

import (
	"github.com/hanifbg/IndonesiaTourismDestination/config"
	"github.com/hanifbg/IndonesiaTourismDestination/internal/repository"
	"github.com/hanifbg/IndonesiaTourismDestination/internal/repository/util"
)

type PlaceService struct {
	placeRepository repository.PlaceRepository
}

func New(config *config.AppConfig, repo *util.RepoWrapper) *PlaceService {
	return &PlaceService{
		placeRepository: repo.PlaceRepo,
	}
}
