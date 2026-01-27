#pragma once

#include <iostream>
#include <string>

class Logger {
public:
    enum Level {
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        FATAL
    };

    static void log(Level level, const std::string& message) {
        const char* levelStr = "";
        switch (level) {
            case DEBUG:   levelStr = "[DEBUG]"; break;
            case INFO:    levelStr = "[INFO]"; break;
            case WARNING: levelStr = "[WARNING]"; break;
            case ERROR:   levelStr = "[ERROR]"; break;
            case FATAL:   levelStr = "[FATAL]"; break;
        }
        std::cout << levelStr << " " << message << std::endl;
    }
};
