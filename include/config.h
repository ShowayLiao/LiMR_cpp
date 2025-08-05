# ifndef CONFIG_H
# define CONFIG_H
# include <string>
# include <iostream>

class Config{
    public:
    Config(const std::string& videoSource, 
            const std::string& StuModelPath, 
            const std::string& TeaModelPath,
            const std::string& imgSource = " ", 
            const std::string& type = "F32",
            int batchSize=1, 
            int inputWidth=224, 
            int inputHeight=224, 
            int outputWidth=224, 
            int outputHeight=224);


    std::string videoSource;
    std::string StuModelPath;
    std::string TeaModelPath;
    std::string imgSource;
    std::string type; // "F32" or "INT8"
    int batchSize;
    int inputWidth;
    int inputHeight;
    int outputWidth;
    int outputHeight;
    
};



#endif // CONFIG_H