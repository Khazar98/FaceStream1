# FaceStream - AI Video Analytics System
# Makefile: C++ TensorRT + Go Fiber/Gin Server

.PHONY: all build run clean rebuild help cpp_lib go_build test run_bg

# Configuration
APP_NAME=video_server
BUILD_DIR=build
CPP_LIB=libprocessor.so

# Default target
all: build

# Build both C++ library and Go application
build: cpp_lib go_build
	@echo "✓ Build complete"

# Build C++ shared library
cpp_lib:
	@echo "[BUILD] C++ shared library..."
	$(MAKE) -C pkg/cpp_processor
	@echo "[BUILD] ✓ C++ library built"

# Build Go application (stub mode - no CGO)
go_build:
	@echo "[BUILD] Go application..."
	mkdir -p $(BUILD_DIR)
	CGO_ENABLED=1 GOOS=linux go build -o $(BUILD_DIR)/$(APP_NAME) ./cmd/server
	@echo "[BUILD] ✓ Go application: $(BUILD_DIR)/$(APP_NAME)"

# Clean build artifacts
clean:
	@echo "[CLEAN] Removing build artifacts..."
	$(MAKE) -C pkg/cpp_processor clean
	rm -rf $(BUILD_DIR)
	@echo "[CLEAN] ✓ Cleanup complete"

# Rebuild
rebuild: clean build
	@echo "✓ Rebuild complete"

# Run the application
run: build
	@echo "[RUN] Starting FaceStream server..."
	@echo "API: http://localhost:8080"
	@echo "Press Ctrl+C to stop"
	./$(BUILD_DIR)/$(APP_NAME)

# Run in background
run_bg: build
	@echo "[RUN] Starting FaceStream in background..."
	./$(BUILD_DIR)/$(APP_NAME) &
	@echo "PID: $$!"

# Test
test:
	@echo "[TEST] Running Go tests..."
	go test ./...
	@echo "✓ Tests passed"

# Help
help:
	@echo "FaceStream Build System"
	@echo "======================="
	@echo "Targets:"
	@echo "  make build       - Build C++ and Go (CGO disabled)"
	@echo "  make run         - Build and run server (foreground)"
	@echo "  make run_bg      - Build and run server (background)"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make rebuild     - Clean and build"
	@echo "  make cpp_lib     - Build C++ only"
	@echo "  make go_build    - Build Go only"
	@echo "  make test        - Run unit tests"
	@echo "  make help        - Show this message"
	@echo ""
	@echo "Production with CGO:"
	@echo "  Install: CUDA 11.x, TensorRT 8.x, OpenCV 4.x"
	@echo "  Then: CGO_ENABLED=1 make build"
