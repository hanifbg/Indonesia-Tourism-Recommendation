package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/go-playground/validator"
	"github.com/hanifbg/IndonesiaTourismDestination/config"
	handlerInit "github.com/hanifbg/IndonesiaTourismDestination/internal/handler/util"
	repoInit "github.com/hanifbg/IndonesiaTourismDestination/internal/repository/util"
	servInit "github.com/hanifbg/IndonesiaTourismDestination/internal/service/util"
	"github.com/labstack/echo"
	"github.com/labstack/echo/middleware"
)

// CustomValidator wraps the validator
type CustomValidator struct {
	validator *validator.Validate
}

// Validate validates the struct
func (cv *CustomValidator) Validate(i interface{}) error {
	return cv.validator.Struct(i)
}

func main() {
	cfg, err := config.GetConfig()
	if err != nil {
		panic(err)
	}

	repo, err := repoInit.New(cfg)
	if err != nil {
		panic(err)
	}

	serv := servInit.New(cfg, repo)
	if err != nil {
		panic(err)
	}

	// Initialize Echo
	e := echo.New()
	e.Logger.SetLevel(4) // INFO level

	// Set custom validator
	e.Validator = &CustomValidator{validator: validator.New()}

	// Configure CORS middleware
	e.Use(middleware.CORSWithConfig(middleware.CORSConfig{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{http.MethodGet, http.MethodPost, http.MethodPut, http.MethodDelete, echo.OPTIONS},
		AllowHeaders:     []string{echo.HeaderOrigin, echo.HeaderContentType, echo.HeaderAccept, echo.HeaderAuthorization, "Ngrok-Skip-Browser-Warning"},
		AllowCredentials: true,
	}))

	// --- health check ---
	e.GET("/health", func(c echo.Context) error {
		return c.String(http.StatusOK, "Go API is healthy!")
	})

	// Initialize handlers
	handlerInit.InitHandler(cfg, e, serv)

	// Start server
	serverAddr := "localhost:8081"
	if cfg.AppPort != 0 {
		serverAddr = fmt.Sprintf(":%d", cfg.AppPort)
	}
	go func() {
		if err := e.Start(serverAddr); err != nil && err != http.ErrServerClosed {
			e.Logger.Fatal("shutting down the server")
		}
	}()

	log.Printf("Server is running at http://%s", serverAddr)

	// Wait for interrupt signal to gracefully shutdown the server with a timeout of 10 seconds.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt)
	<-quit

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := e.Shutdown(ctx); err != nil {
		e.Logger.Fatal(err)
	}
}
