package util

import (
	"net/http"
	"time"

	"github.com/hanifbg/IndonesiaTourismDestination/config"
	"github.com/hanifbg/IndonesiaTourismDestination/internal/repository"
	mlapi "github.com/hanifbg/IndonesiaTourismDestination/internal/repository/ml-api"
	db "github.com/hanifbg/IndonesiaTourismDestination/internal/repository/postgres"
)

type RepoWrapper struct {
	PlaceRepo repository.PlaceRepository
	MLApi     repository.MLAPIRepository
}

func New(config *config.AppConfig) (repoWrapper *RepoWrapper, err error) {

	var dbConnection *db.RepoDatabase

	dbConnection, err = db.Init(config)
	if err != nil {
		return nil, err
	}

	httpClient := &http.Client{
		Timeout: 10 * time.Second,
	}

	mlApi := mlapi.New(config, httpClient)

	repoWrapper = &RepoWrapper{
		PlaceRepo: dbConnection,
		MLApi:     mlApi.MLApi,
	}

	return
}
