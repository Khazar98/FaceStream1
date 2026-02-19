# FaceStream C++ Processor Makefile (TensorRT)
# Compiles C++ into shared library for face detection

# Paths
# CUDA_PATH ?= /usr/local/cuda  <-- REMOVED
OPENCV_PATH ?= /usr/local

# Compiler
CXX = g++

# Output
OUTPUT = libprocessor.so
OBJECTS = processor.o tracker.o

# Compiler flags
# Added -I for CUDA and TensorRT headers
CXXFLAGS = -fPIC -shared -std=c++17 -Wall -Wextra -O3 \
           -I. \
           -I/usr/include \
           -I/usr/include/opencv4

# Linker flags
# Added -lnvinfer (TensorRT) and -lcudart (CUDA Runtime)
LDFLAGS = -L/usr/lib/x86_64-linux-gnu -L$(OPENCV_PATH)/lib \
          -lnvinfer -lcudart \
          -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lstdc++ -lm \
          -Wl,-rpath,/usr/lib/x86_64-linux-gnu

# Build targets
all: $(OUTPUT)

$(OUTPUT): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "[BUILD] Compiled: $@"

processor.o: processor.cpp processor.h tracker.h
	$(CXX) $(CXXFLAGS) -c processor.cpp -o $@
	@echo "[BUILD] Compiled object: $@"

tracker.o: tracker.cpp tracker.h
	$(CXX) $(CXXFLAGS) -c tracker.cpp -o $@
	@echo "[BUILD] Compiled object: $@"

# Cleanup
clean:
	rm -f $(OBJECTS) $(OUTPUT)
	@echo "[CLEAN] Cleanup complete"

# Rebuild
rebuild: clean all

.PHONY: all clean rebuild
