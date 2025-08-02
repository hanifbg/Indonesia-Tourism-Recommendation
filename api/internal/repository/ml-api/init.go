package mlapi

import (
	"net/http"

	"github.com/hanifbg/IndonesiaTourismDestination/config"
)

type ApiWrapper struct {
	MLApi *MLApiClient
}

type MLApiClient struct {
	Client *http.Client
	Config *config.AppConfig
}

func New(config *config.AppConfig, client *http.Client) *ApiWrapper {
	return &ApiWrapper{
		MLApi: &MLApiClient{
			Client: client,
			Config: config,
		},
	}
}
