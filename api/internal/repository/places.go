package repository

import "github.com/hanifbg/IndonesiaTourismDestination/internal/model/entity"

type PlaceRepository interface {
	GetByID(id int) (*entity.Place, error)
	GetAllPlaces() ([]entity.Place, error)
}
