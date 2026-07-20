#pragma once

struct RenderResolutionOption {
    const char* label;
    int width;
    int height;
};

inline constexpr RenderResolutionOption kRenderResolutionOptions[] = {
    {"224x224", 224, 224},
    {"256x256", 256, 256},
    {"448x448", 448, 448},
    {"512x512", 512, 512},
    {"1024x1024", 1024, 1024},
};

inline constexpr int kRenderResolutionOptionCount =
    static_cast<int>(sizeof(kRenderResolutionOptions) / sizeof(kRenderResolutionOptions[0]));
