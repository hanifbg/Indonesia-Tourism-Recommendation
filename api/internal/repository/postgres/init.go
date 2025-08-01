package postgres

import (
	"fmt"

	"github.com/hanifbg/IndonesiaTourismDestination/config"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

type RepoDatabase struct {
	DB *gorm.DB
}

func Init(config *config.AppConfig) (*RepoDatabase, error) {
	repo := &RepoDatabase{}
	db, err := getConnection(config)
	if err != nil {
		return nil, err
	}

	repo.DB = db
	return repo, nil
}

func getConnection(config *config.AppConfig) (*gorm.DB, error) {
	dsn := fmt.Sprintf("host=%s user=%s password=%s dbname=%s port=%d sslmode=%s",
		config.DbHost,
		config.DbUser,
		config.DbPassword,
		config.DbName,
		config.DbPort,
		config.DbSSLMode,
	)

	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		return nil, err
	}

	// Enable uuid-ossp extension
	if err := db.Exec("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"").Error; err != nil {
		return nil, fmt.Errorf("failed to enable uuid-ossp extension: %v", err)
	}

	return db, nil
}
