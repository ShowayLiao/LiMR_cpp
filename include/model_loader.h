#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <string>
#include <vector>
#include <stdexcept>

/**
 * @brief Model file type enumeration
 */
enum class ModelFileType {
    UNKNOWN,  ///< Unknown type
    ONNX,     ///< ONNX format
    ENGINE    ///< TensorRT ENGINE format
};

/**
 * @brief Model loader exception class
 */
class ModelLoaderException : public std::runtime_error {
public:
    explicit ModelLoaderException(const std::string& message) : std::runtime_error(message) {}
};

/**
 * @brief Model loader class, responsible for detecting model file types and performing corresponding operations
 */
class ModelLoader {
public:
    /**
     * @brief Constructor
     * @param cacheDir Model cache directory, default to .cache folder in current directory
     */
    explicit ModelLoader(const std::string& cacheDir = "./.cache");

    /**
     * @brief Destructor
     */
    ~ModelLoader();

    /**
     * @brief Detect model file type
     * @param filePath Model file path
     * @return Model file type
     * @throws ModelLoaderException Throws when file does not exist or cannot be read
     */
    ModelFileType detectModelType(const std::string& filePath);

    /**
     * @brief Process model file, perform corresponding operations based on type
     * @param filePath Model file path
     * @param inputWidth Model input width
     * @param inputHeight Model input height
     * @param batchSize Batch size
     * @param precision Precision type, e.g., "F32", "F16"
     * @return Final ENGINE file path used
     * @throws ModelLoaderException Throws when processing fails
     */
    std::string processModel(const std::string& filePath, 
                             int inputWidth = 224, 
                             int inputHeight = 224, 
                             int batchSize = 1, 
                             const std::string& precision = "F32");

    /**
     * @brief Check if file exists
     * @param filePath File path
     * @return Whether the file exists
     */
    static bool fileExists(const std::string& filePath);

    /**
     * @brief Set whether to enable ONNX conversion functionality
     * @param enable Whether to enable
     */
    void setOnnxConversionEnabled(bool enable);

    /**
     * @brief Check if ONNX conversion functionality is enabled
     * @return Whether enabled
     */
    bool isOnnxConversionEnabled() const;

private:
    /**
     * @brief Detect ONNX format by file header
     * @param filePath File path
     * @return Whether it is ONNX format
     */
    bool isOnnxByHeader(const std::string& filePath);

    /**
     * @brief Detect ENGINE format by file header
     * @param filePath File path
     * @return Whether it is ENGINE format
     */
    bool isEngineByHeader(const std::string& filePath);

    /**
     * @brief Convert ONNX model to ENGINE format
     * @param onnxPath ONNX file path
     * @param enginePath Output ENGINE file path
     * @param inputWidth Model input width
     * @param inputHeight Model input height
     * @param batchSize Batch size
     * @param precision Precision type
     * @throws ModelLoaderException Throws when conversion fails
     */
    void convertOnnxToEngine(const std::string& onnxPath, 
                            const std::string& enginePath, 
                            int inputWidth, 
                            int inputHeight, 
                            int batchSize,
                            const std::string& precision);

    /**
     * @brief Generate cache file path
     * @param originalPath Original file path
     * @param precision Precision type
     * @return Cache file path
     */
    std::string generateCachePath(const std::string& originalPath, const std::string& precision);

    std::string cacheDir_;  ///< Model cache directory
    bool enableOnnxConversion_;  ///< Whether ONNX conversion functionality is enabled
};

#endif // MODEL_LOADER_H
