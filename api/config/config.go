package config

import (
	"fmt"
	"os"
	"sync"

	"github.com/spf13/viper"
)

type AppConfig struct {
	DbHost     string
	DbUser     string
	DbPassword string
	DbName     string
	DbPort     int
	DbSSLMode  string
	AppPort    int
}

var (
	lock      = &sync.Mutex{}
	appConfig *AppConfig
)

func GetConfig() (*AppConfig, error) {
	if appConfig != nil {
		return appConfig, nil
	}

	lock.Lock()
	defer lock.Unlock()

	if appConfig != nil {
		return appConfig, nil
	}

	appConfig, err := initConfig()
	return appConfig, err
}

func initConfig() (*AppConfig, error) {
	var finalConfig AppConfig

	viper.AddConfigPath(".")
	viper.AddConfigPath("./config")
	viper.SetConfigName("app.config")
	viper.SetConfigType("json")
	err := viper.ReadInConfig()

	finalConfig.AppPort = viper.GetInt("server.port")
	finalConfig.DbHost = viper.GetString("database.host")
	finalConfig.DbPort = viper.GetInt("database.port")
	finalConfig.DbUser = viper.GetString("database.username")
	finalConfig.DbPassword = viper.GetString("database.password")
	finalConfig.DbName = viper.GetString("database.dbname")
	finalConfig.DbSSLMode = viper.GetString("database.sslmode")
	if err != nil {
		return nil, err
	}

	fmt.Printf("Using config file: %s\n\n", viper.ConfigFileUsed())

	return &finalConfig, nil
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvIntOrDefault(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := fmt.Sscanf(value, "%d"); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvBoolOrDefault(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		return value == "true" || value == "1" || value == "yes"
	}
	return defaultValue
}
