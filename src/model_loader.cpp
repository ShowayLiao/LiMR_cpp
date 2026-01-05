#include "model_loader.h"
#include "utils.h"
#include <fstream>
#include <regex>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <sys/stat.h>
#include <direct.h>
#include <time.h>
#include <string.h>
#include <errno.h>

#ifdef _WIN32
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir(path, 0755)
#endif

const std::vector<unsigned char> ONNX_MAGIC = {0x4F, 0x4E, 0x4E, 0x58};
const std::vector<unsigned char> ENGINE_MAGIC = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

ModelLoader::ModelLoader(const std::string& cacheDir)
    : cacheDir_(cacheDir),
      enableOnnxConversion_(true) {
    int result = MKDIR(cacheDir_.c_str());
    if (result != 0 && errno != EEXIST) {
        std::cerr << "[WARNING] Failed to create cache directory: " << strerror(errno) << std::endl;
    }
}

ModelLoader::~ModelLoader() {
}

ModelFileType ModelLoader::detectModelType(const std::string& filePath) {
    if (!fileExists(filePath)) {
        throw ModelLoaderException("File not found: " + filePath);
    }

    std::string ext;
    size_t dotPos = filePath.find_last_of(".");
    if (dotPos != std::string::npos && dotPos < filePath.size() - 1) {
        ext = filePath.substr(dotPos + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    std::cout << "[DEBUG] File: " << filePath << ", Extension: " << ext << std::endl;

    if (ext == "onnx") {
        std::cout << "[DEBUG] Extension is ONNX, attempting direct ONNX processing" << std::endl;
        return ModelFileType::ONNX;
    } else if (ext == "engine") {
        if (isEngineByHeader(filePath)) {
            return ModelFileType::ENGINE;
        }
    }

    std::cout << "[DEBUG] Checking file headers directly..." << std::endl;
    if (isOnnxByHeader(filePath)) {
        std::cout << "[DEBUG] Direct header check: ONNX" << std::endl;
        return ModelFileType::ONNX;
    } else if (isEngineByHeader(filePath)) {
        std::cout << "[DEBUG] Direct header check: ENGINE" << std::endl;
        return ModelFileType::ENGINE;
    }

    std::cout << "[DEBUG] Unknown file type" << std::endl;
    return ModelFileType::UNKNOWN;
}

std::string ModelLoader::processModel(const std::string& filePath,
                                     int inputWidth,
                                     int inputHeight,
                                     int batchSize,
                                     const std::string& precision) {
    ModelFileType type = detectModelType(filePath);
    std::string enginePath;

    switch (type) {
        case ModelFileType::ENGINE:
            std::cout << "[INFO] Model is already in ENGINE format, using directly: " << filePath << std::endl;
            return filePath;

        case ModelFileType::ONNX: {
            if (!enableOnnxConversion_) {
                throw ModelLoaderException("ONNX conversion is disabled, please provide ENGINE file directly");
            }

            enginePath = generateCachePath(filePath, precision);

            if (fileExists(enginePath)) {
                std::cout << "[INFO] Found cached ENGINE file: " << enginePath << std::endl;
                return enginePath;
            }

            std::cout << "[INFO] Converting ONNX to ENGINE format..." << std::endl;
            std::cout << "[INFO] Input: " << filePath << std::endl;
            std::cout << "[INFO] Output: " << enginePath << std::endl;
            std::cout << "[INFO] Precision: " << precision << std::endl;
            std::cout << "[INFO] Input Size: " << inputWidth << "x" << inputHeight << std::endl;
            std::cout << "[INFO] Batch Size: " << batchSize << std::endl;

            try {
                build_engine(filePath, enginePath, batchSize, inputWidth, inputHeight, inputWidth, inputHeight);
                std::cout << "[INFO] ONNX to ENGINE conversion completed successfully" << std::endl;
                return enginePath;
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] ONNX to ENGINE conversion failed: " << e.what() << std::endl;
                throw ModelLoaderException("ONNX to ENGINE conversion failed: " + std::string(e.what()));
            }
        }

        default:
            throw ModelLoaderException("Unsupported model file type: " + filePath);
    }
}

bool ModelLoader::fileExists(const std::string& filePath) {
    struct stat buffer;
    if (stat(filePath.c_str(), &buffer) != 0) {
        std::cerr << "[DEBUG] File does not exist: " << filePath << std::endl;
        return false;
    }
    std::cout << "[DEBUG] File exists: " << filePath << std::endl;
    return (buffer.st_mode & S_IFREG) != 0;
}

void ModelLoader::setOnnxConversionEnabled(bool enable) {
    enableOnnxConversion_ = enable;
}

bool ModelLoader::isOnnxConversionEnabled() const {
    return enableOnnxConversion_;
}

bool ModelLoader::isOnnxByHeader(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[DEBUG] Failed to open file for ONNX header check: " << filePath << std::endl;
        return false;
    }

    std::vector<unsigned char> header(ONNX_MAGIC.size());
    file.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));

    if (!file) {
        std::cerr << "[DEBUG] Failed to read header for ONNX check: " << filePath << std::endl;
        return false;
    }

    bool match = (header == ONNX_MAGIC);
    std::cout << "[DEBUG] ONNX header check for " << filePath << ": " << (match ? "MATCH" : "NO MATCH") << std::endl;
    std::cout << "[DEBUG] Expected: ONNX, Got: ";
    for (unsigned char c : header) {
        std::cout << std::hex << static_cast<int>(c) << " ";
    }
    std::cout << std::dec << std::endl;

    return match;
}

bool ModelLoader::isEngineByHeader(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[DEBUG] Failed to open file for ENGINE header check: " << filePath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    if (fileSize < 1024 * 1024) {
        std::cout << "[DEBUG] File too small for ENGINE: " << filePath << ", Size: " << fileSize << " bytes" << std::endl;
        return false;
    }

    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> header(ENGINE_MAGIC.size());
    file.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));

    if (!file) {
        std::cerr << "[DEBUG] Failed to read header for ENGINE check: " << filePath << std::endl;
        return false;
    }

    bool isEngine = (header == ENGINE_MAGIC);
    std::cout << "[DEBUG] ENGINE header check for " << filePath << ": " << (isEngine ? "MATCH" : "NO MATCH") << std::endl;
    std::cout << "[DEBUG] File size: " << fileSize << " bytes" << std::endl;

    return isEngine;
}

void ModelLoader::convertOnnxToEngine(const std::string& onnxPath,
                                     const std::string& enginePath,
                                     int inputWidth,
                                     int inputHeight,
                                     int batchSize,
                                     const std::string& precision) {
    try {
        build_engine(onnxPath, enginePath, batchSize, inputWidth, inputHeight, inputWidth, inputHeight);
    } catch (const std::exception& e) {
        throw ModelLoaderException("ONNX to ENGINE conversion failed: " + std::string(e.what()));
    }
}

std::string ModelLoader::generateCachePath(const std::string& originalPath, const std::string& precision) {
    size_t lastDot = originalPath.find_last_of(".");
    std::string filename = (lastDot != std::string::npos) ? originalPath.substr(0, lastDot) : originalPath;
    size_t lastSlash = filename.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        filename = filename.substr(lastSlash + 1);
    }

    struct stat fileStat;
    if (stat(originalPath.c_str(), &fileStat) != 0) {
        time_t now = time(nullptr);
        std::ostringstream oss;
        oss << cacheDir_ << "/" << filename << "_" << precision << "_" << now << ".engine";
        return oss.str();
    }

    uintmax_t fileSize = fileStat.st_size;
    time_t modTime = fileStat.st_mtime;

    std::ostringstream oss;
    oss << cacheDir_ << "/" << filename << "_" << precision << "_" << fileSize << "_" << modTime << ".engine";

    return oss.str();
}
