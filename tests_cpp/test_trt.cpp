#include <NvInfer.h>
#include <iostream>
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        std::cout << "[TRT] " << msg << std::endl;
    }
};
Logger gLogger;
int main() {
    std::cout << "Creating TRT..." << std::endl;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        std::cout << "Failed!" << std::endl;
        return 1;
    }
    std::cout << "Success!" << std::endl;
    delete runtime;
    return 0;
}
