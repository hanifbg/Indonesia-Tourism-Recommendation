package service

import "github.com/hanifbg/IndonesiaTourismDestination/internal/model/entity"

type PlaceService interface {
	GetPlaceByID(id int) (*entity.Place, error)
}
