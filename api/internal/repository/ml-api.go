package repository

import "github.com/hanifbg/IndonesiaTourismDestination/internal/model/entity"

type MLAPIRepository interface {
	GetRecommendations(request *entity.RecommendationRequestML) (entity.RecommendationResponseML, error)
}
