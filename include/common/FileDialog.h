#pragma once

#include <cstddef>

namespace common {

bool copySelectedPath(char* destination,
                      std::size_t destination_size,
                      const char* selected_path);

bool selectFile(char* destination,
                std::size_t destination_size,
                const char* filter);

}  // namespace common
