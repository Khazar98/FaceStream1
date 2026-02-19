// Quick C++ test to print tensor shapes
#include <iostream>
#include <NvInfer.h>
#include <fstream>
#include <vector>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;

int main() {
    // Load engine
    std::ifstream file("./models/yolov12n-face.engine", std::ios::binary);
    if (!file.good()) {
        std::cerr << "Engine file not found!" << std::endl;
        return 1;
    }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();
    
    // Create runtime and deserialize
    auto runtime = nvinfer1::createInferRuntime(gLogger);
    auto engine = runtime->deserializeCudaEngine(buffer.data(), size);
    
    if (!engine) {
        std::cerr << "Failed to deserialize engine!" << std::endl;
        return 1;
    }
    
    // Print tensor shapes
    std::cout << "=== Engine Tensor Shapes ===" << std::endl;
    
    auto input_dims = engine->getTensorShape("images");
    std::cout << "Input 'images' shape: [";
    for (int i = 0; i < input_dims.nbDims; i++) {
        std::cout << input_dims.d[i];
        if (i < input_dims.nbDims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    auto output_dims = engine->getTensorShape("output0");
    std::cout << "Output 'output0' shape: [";
    for (int i = 0; i < output_dims.nbDims; i++) {
        std::cout << output_dims.d[i];
        if (i < output_dims.nbDims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Calculate total elements
    int output_elements = 1;
    for (int i = 0; i < output_dims.nbDims; i++) {
        output_elements *= output_dims.d[i];
    }
    std::cout << "Total output elements: " << output_elements << std::endl;
    std::cout << "Total output bytes (FP32): " << (output_elements * 4) << std::endl;
    
    delete engine;
    delete runtime;
    
    return 0;
}
