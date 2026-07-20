#include "dashboard_render_options.h"

#include <cassert>

int main() {
    assert(kRenderResolutionOptionCount == 5);
    assert(kRenderResolutionOptions[0].width == 224);
    assert(kRenderResolutionOptions[1].width == 256);
    assert(kRenderResolutionOptions[2].width == 448);
    assert(kRenderResolutionOptions[3].width == 512);
    assert(kRenderResolutionOptions[4].width == 1024);
    return 0;
}
