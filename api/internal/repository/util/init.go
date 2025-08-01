package util

import (
	"github.com/hanifbg/IndonesiaTourismDestination/config"
	"github.com/hanifbg/IndonesiaTourismDestination/internal/repository"
	db "github.com/hanifbg/IndonesiaTourismDestination/internal/repository/postgres"
)

type RepoWrapper struct {
	PlaceRepo repository.PlaceRepository
}

func New(config *config.AppConfig) (repoWrapper *RepoWrapper, err error) {

	var dbConnection *db.RepoDatabase

	dbConnection, err = db.Init(config)
	if err != nil {
		return nil, err
	}

	// httpClient := &http.Client{
	// 	Timeout: 10 * time.Second,
	// }

	repoWrapper = &RepoWrapper{
		PlaceRepo: dbConnection,
	}

	return
}
