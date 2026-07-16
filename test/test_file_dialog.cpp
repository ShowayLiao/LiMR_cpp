#include <cassert>
#include <cstring>

#include "common/FileDialog.h"

int main() {
    char selected[64] = "old-path";
    assert(common::copySelectedPath(selected, sizeof(selected), "C:\\input\\part.png"));
    assert(std::strcmp(selected, "C:\\input\\part.png") == 0);

    char cancelled[64] = "keep-this";
    assert(!common::copySelectedPath(cancelled, sizeof(cancelled), nullptr));
    assert(std::strcmp(cancelled, "keep-this") == 0);

    char too_small[8] = "current";
    assert(!common::copySelectedPath(too_small, sizeof(too_small), "C:\\input\\part.png"));
    assert(std::strcmp(too_small, "current") == 0);
    return 0;
}
