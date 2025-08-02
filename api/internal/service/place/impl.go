package place

import "github.com/hanifbg/IndonesiaTourismDestination/internal/model/entity"

func (p *PlaceService) GetPlaceByID(id int) (*entity.Place, error) {
	result, err := p.placeRepository.GetByID(id)
	if err != nil {
		return nil, err
	}
	return result, nil
}

func (p *PlaceService) GetRecommendations(request *entity.RecommendationRequestML) (entity.RecommendationResponseML, error) {
	return p.mlApi.GetRecommendations(request)
}
