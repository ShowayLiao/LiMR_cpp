# Implement Texture Management and Main Loop Integration

## Overview

This plan implements the requested changes to the Dashboard and main.cpp files to properly handle texture management and integrate the Dashboard into the main loop with CUDA Interop support.

## Changes to Dashboard.h

1. **Add InitResources method** - For texture ID generation
2. **Add UpdateData method** - For updating texture content
3. **Add GetTextureIDs method** - For retrieving texture IDs
4. **Add member variables** - For tracking resource initialization status

## Changes to Dashboard.cpp

1. **Implement InitResources** - Generate texture IDs once during initialization
2. **Implement UpdateData** - Update texture content or just bind IDs if using CUDA Interop
3. **Modify Render** - Remove glGenTextures calls, use initialized IDs
4. **Modify DrawMainView** - Implement proper texture drawing with CUDA Interop support
5. **Update destructor** - Ensure proper cleanup of resources

## Changes to main.cpp

1. **Add Dashboard initialization** - Create and initialize Dashboard
2. **Add resource initialization** - Call Dashboard::InitResources after OpenGL context creation
3. **Add texture ID registration** - Register Dashboard's texture IDs with Pipeline
4. **Modify main loop** - Integrate Dashboard::UpdateData and Dashboard::Render
5. **Add cleanup** - Ensure proper cleanup of Dashboard resources

## Key Implementation Details

1. **Texture Management**: Generate texture IDs in InitResources, update content in UpdateData
2. **CUDA Interop Support**: Handle both CPU and GPU texture data scenarios
3. **Rendering**: Implement three-layer rendering (Original → Heatmap → Mask)
4. **Integration**: Properly register texture IDs with Pipeline for CUDA-based updates
5. **Performance**: Avoid glGenTextures in render loop, minimize GPU-CPU data transfers

## Expected Outcome

* Proper texture management with CUDA Interop support

* Efficient rendering pipeline

* Clean integration of Dashboard into main loop

* Support for both CPU and GPU texture data scenarios

