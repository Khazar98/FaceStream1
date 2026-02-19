package main

import (
	"io"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"video-analytics-system/internal/kafka"

	"github.com/gin-gonic/gin"
)

func main() {
	// Set up file logging to server.log
	logFile, err := os.OpenFile("server.log", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Failed to open log file: %v", err)
	}
	defer logFile.Close()

	// Create multi-writer to write both to console and file
	multiWriter := io.MultiWriter(os.Stdout, logFile)
	log.SetOutput(multiWriter)
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	log.Println("========================================")
	log.Println("FaceStream AI Video Analytics System")
	log.Println("========================================")

	// Kafka Producer-i başladırıq.
	// Env dəyişənlərindən oxuyuruq, yoxdursa default dəyərləri götürürük.
	kafkaBrokersStr := os.Getenv("KAFKA_BROKERS")
	var kafkaBrokers []string
	if kafkaBrokersStr != "" {
		for _, b := range strings.Split(kafkaBrokersStr, ",") {
			kafkaBrokers = append(kafkaBrokers, strings.TrimSpace(b))
		}
	} else {
		kafkaBrokers = []string{"10.13.3.100:9092", "10.13.3.99:9092", "10.13.3.101:9092"}
	}

	log.Printf("Kafka Brokers: %v", kafkaBrokers)

	producer, err := kafka.NewProducer(kafkaBrokers, "detectiontopic", "detectionMega")
	if err != nil {
		log.Fatalf("Failed to create Kafka producer: %v", err)
	}
	defer producer.Close()

	// C++ AI Nüvəsini başladırıq. Callback funksiyasını Cgo vasitəsilə ötürürük.
	// `InitializeAIProcessor` funksiyası cgo_bridge.go faylında təyin ediləcək.
	// Modelin yolu: Env-dən oxuyuruq və ya default ./models/yolov12n-face.engine
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "./models/yolov12n-face.engine"
	}

	// Ensure model exists
	absModelPath, err := filepath.Abs(modelPath)
	if err != nil {
		log.Fatalf("Failed to get absolute path for model: %v", err)
	}

	if _, err := os.Stat(absModelPath); err != nil {
		log.Fatalf("Model not found at %s: %v", absModelPath, err)
	}

	log.Printf("Using AI model: %s", absModelPath)

	if err := InitializeAIProcessor(modelPath, producer); err != nil {
		log.Fatalf("Failed to initialize AI processor with model %s: %v", modelPath, err)
	}
	// Proqram bitdikdə C++ resurslarını təmizləmək üçün defer istifadə edirik.
	defer ShutdownAIProcessor()

	// Goroutine-ləri idarə etmək üçün Worker Manager-i yaradırıq.
	manager := NewWorkerManager()

	// Gin router-i qururuq.
	router := gin.Default()

	// API endpoint-lərini və onların handler-lərini təyin edirik.
	api := NewAPI(manager)
	cameraGroup := router.Group("/camera")
	{
		cameraGroup.POST("/start", api.StartCameraHandler)
		cameraGroup.POST("/stop", api.StopCameraHandler)
		cameraGroup.GET("/status", api.GetStatusHandler)
	}

	// Serveri ayrı bir goroutine-də başladırıq ki, əsas thread bloklanmasın.
	go func() {
		log.Println("Server starting on port :8050")
		if err := router.Run(":8050"); err != nil {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Graceful shutdown üçün siqnalları gözləyirik (Ctrl+C).
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down server...")

	// Bütün aktiv worker-ləri dayandırırıq.
	manager.StopAllWorkers()
	log.Println("All camera workers stopped.")
}
