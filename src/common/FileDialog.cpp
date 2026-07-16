#include "common/FileDialog.h"

#include <cstring>

#include <windows.h>
#include <commdlg.h>

namespace common {

bool copySelectedPath(char* destination,
                      std::size_t destination_size,
                      const char* selected_path) {
    if (destination == nullptr || destination_size == 0 || selected_path == nullptr) {
        return false;
    }

    const std::size_t selected_size = std::strlen(selected_path) + 1;
    if (selected_size > destination_size) {
        return false;
    }

    std::memcpy(destination, selected_path, selected_size);
    return true;
}

bool selectFile(char* destination,
                std::size_t destination_size,
                const char* filter) {
    char selected_path[1024] = {};
    OPENFILENAMEA dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.lpstrFile = selected_path;
    dialog.nMaxFile = static_cast<DWORD>(sizeof(selected_path));
    dialog.lpstrFilter = filter;
    dialog.nFilterIndex = 1;
    dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameA(&dialog) == FALSE) {
        return false;
    }
    return copySelectedPath(destination, destination_size, selected_path);
}

}  // namespace common
