package postgres

import "github.com/hanifbg/IndonesiaTourismDestination/internal/model/entity"

func (r *RepoDatabase) GetAllPlaces() ([]entity.Place, error) {
	var places []entity.Place
	err := r.DB.Find(&places).Error
	if err != nil {
		return nil, err
	}
	return places, nil
}

func (r *RepoDatabase) GetByID(id int) (*entity.Place, error) {
	var place entity.Place
	err := r.DB.First(&place, id).Error
	if err != nil {
		return nil, err
	}
	return &place, nil
}
