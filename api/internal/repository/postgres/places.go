package postgres

import (
	"github.com/hanifbg/IndonesiaTourismDestination/internal/model/entity"
)

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
	tx := r.DB.Raw("SELECT place_id, place_name, description, category, city, price, rating, time_minutes, coordinate, lat, long FROM places WHERE place_id = $1", id)
	err := tx.Scan(&place).Error
	if err != nil {
		return nil, err
	}
	return &place, nil
}
