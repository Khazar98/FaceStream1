package main

import (
	"log"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// API struct-u worker manager-dən asılılığı saxlayır.
type API struct {
	manager *WorkerManager
}

// Yeni bir API obyekti yaratmaq üçün konstruktor.
func NewAPI(manager *WorkerManager) *API {
	return &API{manager: manager}
}

// POST /camera/start üçün JSON body-ni təmsil edən struct.
type StartRequest struct {
	RTSPLink string `json:"rtsp_link" binding:"required"`
	CameraID string `json:"camera_id" binding:"required,uuid"`
}

// POST /camera/stop üçün JSON body-ni təmsil edən struct.
// Həm camera_id, həm də worker_id qəbul edə bilər.
type StopRequest struct {
	CameraID string `json:"camera_id"`
	WorkerID string `json:"worker_id"`
}

// StartCameraHandler /camera/start endpoint-ini idarə edir.
func (a *API) StartCameraHandler(c *gin.Context) {
	var req StartRequest
	// Gələn JSON-u yoxlayırıq.
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body: " + err.Error()})
		return
	}

	// Worker Manager vasitəsilə yeni bir RTSP emalı goroutine-i başladırıq.
	workerID, err := a.manager.StartWorker(req.CameraID, req.RTSPLink)
	if err != nil {
		// Əgər həmin ID ilə kamera artıq işləyirsə, xəta qaytarırıq.
		if err == ErrCameraAlreadyRunning {
			c.JSON(http.StatusConflict, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to start camera worker: " + err.Error()})
		return
	}

	log.Printf("Started camera %s with worker ID %d", req.CameraID, workerID)
	// Uğurlu cavab qaytarırıq.
	c.JSON(http.StatusOK, gin.H{
		"worker_id": workerID,
		"status":    "running",
	})
}

// StopCameraHandler /camera/stop endpoint-ini idarə edir.
func (a *API) StopCameraHandler(c *gin.Context) {
	var req StopRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body: " + err.Error()})
		return
	}

	var err error
	if req.CameraID != "" {
		// UUID formatının düzgünlüyünü yoxlayırıq.
		if _, parseErr := uuid.Parse(req.CameraID); parseErr != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid camera_id format"})
			return
		}
		err = a.manager.StopWorkerByCameraID(req.CameraID)
	} else if req.WorkerID != "" {
		id, convErr := strconv.ParseInt(req.WorkerID, 10, 64)
		if convErr != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "worker_id must be an integer"})
			return
		}
		err = a.manager.StopWorkerByWorkerID(id)
	} else {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Either camera_id or worker_id must be provided"})
		return
	}

	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	log.Printf("Stopped camera worker (request: %+v)", req)
	c.JSON(http.StatusOK, gin.H{"status": "stopped"})
}

// GetStatusHandler /camera/status endpoint-ini idarə edir.
func (a *API) GetStatusHandler(c *gin.Context) {
	// Worker Manager-dən aktiv worker-lərin statusunu alırıq.
	status := a.manager.GetStatus()
	c.JSON(http.StatusOK, status)
}
