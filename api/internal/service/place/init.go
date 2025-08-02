package place

import (
	"github.com/hanifbg/IndonesiaTourismDestination/config"
	"github.com/hanifbg/IndonesiaTourismDestination/internal/repository"
	"github.com/hanifbg/IndonesiaTourismDestination/internal/repository/util"
)

type PlaceService struct {
	placeRepository repository.PlaceRepository
	mlApi           repository.MLAPIRepository
}

func New(config *config.AppConfig, repo *util.RepoWrapper) *PlaceService {
	return &PlaceService{
		placeRepository: repo.PlaceRepo,
		mlApi:           repo.MLApi,
	}
}
