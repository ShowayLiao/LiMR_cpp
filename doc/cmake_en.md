<div align="center">

<h1>CMake Configuration Guide</h1>

</div>

# 📌 C++ Code Generation Workflow
## Preprocessing
1. Preprocessing is the first step in C/C++ compilation, handling macro replacement, file inclusion, and conditional compilation.  
2. `#define`: Defines macros.  
3. `#include`: Includes header files.  
4. `#ifdef`, `#ifndef`, `#endif`: Conditional compilation.  
5. `#pragma`: Passes special instructions to the compiler (often for platform-specific optimizations).  

## Compilation
1. Converts preprocessed source code into assembly code, generating `.s` files.  

## Assembly
1. Translates assembly code into machine code, producing `.o` files. These are not executable yet (missing library links).  

## Linking
1. Combines object files into final executables or libraries. Two types:  
### Static Linking
1. Occurs at compile time. Embeds all code/data into a single executable.  
   - Example: `add_library(pipeline STATIC pipeline.cpp)` generates `pipeline.lib`, embedded into `main.exe`.  
   - Drawback: Redundant copies if multiple programs use the same library.  

### Dynamic Linking
1. Links shared libraries (`.so`/`.dll`) at runtime.  

<details style="color:rgb(128,128,128)">
<summary>Library Types</summary>

#### Static Library (STATIC)
1. Output: `.a` (Linux) / `.lib` (Windows).  
2. Fully copied into executables. Increases file size but no runtime dependencies.

```cmake
add_library(mylib STATIC src/file1.cpp src/file2.cpp)
```
#### Shared Library (SHARED)
1. Output: `.so` (Linux) / `.dll` (Windows).  
2. Dynamically loaded at runtime. Saves memory via shared access.
```cmake
add_library(mylib SHARED src/file1.cpp src/file2.cpp)
```

#### Module Library (MODULE)
1. Similar to shared libraries but loaded dynamically via `dlopen()` (Linux) or `LoadLibrary()` (Windows).

```cmake
add_library(mymodule MODULE src/module.cpp)
```

#### Object Library (OBJECT)
1. Compiles sources to `.o`/`.obj` files without creating a full library. Reusable across targets.

```cmake
add_library(myobjects OBJECT src/file1.cpp src/file2.cpp)

add_executable(myapp $<TARGET_OBJECTS:myobjects> src/main.cpp)
```

</details>

# 📌 Dynamic Libraries
1. Dynamic libraries generate two files:  
   - **Import Library** (`.lib` or `.dll.a`): Contains function definitions.  
   - **Dynamic Library** (`.dll`): Holds complete function code.  
2. Most third-party libraries use this format.  

## Using Dynamic Libraries
### Implicit Linking
1. Loads the DLL at program startup. Requires:  
   - Header files (`.h`).  
   - Import library (`.lib`).  
   - DLL in the executable's directory or `PATH`.  

### Explicit Linking
1. Manually load/unload libraries using system APIs:

```cpp

# include <windows.h>
# include <iostream>
typedef int (*AddFunc)(int, int);

int main() {

    HINSTANCE hDll = LoadLibrary("MathLib.dll");

    if (hDll) {

    AddFunc add = (AddFunc)GetProcAddress(hDll, "add");

    if (add) {

    std::cout << add(2, 3) << std::endl; // Outputs 5

    }

    FreeLibrary(hDll);

    }

    return 0;

}

```

# 📌 Common CMake Variables
## Global Variables
1. `${CMAKE_SOURCE_DIR}`: Root directory of the project (contains top-level `CMakeLists.txt`).  
2. `${CMAKE_BINARY_DIR}`: Build directory (e.g., `./build`).  
3. `${CMAKE_CURRENT_SOURCE_DIR}`: Directory of the current `CMakeLists.txt`.  
4. `${CMAKE_CURRENT_BINARY_DIR}`: Build directory for the current `CMakeLists.txt`.  
5. `${CMAKE_MODULE_PATH}`: Paths for custom CMake modules.  
6. `${CMAKE_INSTALL_PREFIX}`: Installation root directory.  
7. `${CMAKE_LIBRARY_OUTPUT_DIRECTORY}` / `${CMAKE_RUNTIME_OUTPUT_DIRECTORY}`: Output paths for libraries/executables.  
8. `${CMAKE_BUILD_TYPE}`: Build configuration (e.g., `Debug`, `Release`).  
9. `EXECUTABLE_OUTPUT_PATH`: Output path for executables.  
10. `LIBRARY_OUTPUT_PATH`: Output path for libraries.  

## Custom Variables
1. **Implicit**: Defined via `PROJECT()` (e.g., `<project>_BINARY_DIR`).  
2. **Explicit**: Defined via `SET()` (e.g., `SET(HELLO_SRC main.c)` → `${HELLO_SRC}`).  

## Compiler Variables
1. `find_program`: Locates executables.  
2. `CMAKE_CXX_COMPILER`: Path to the C++ compiler.  
3. `CMAKE_BUILD_TYPE`: Sets optimization/debug levels (override via `-DCMAKE_BUILD_TYPE=<type>`).  

# 📌 Core CMake Commands
## 1. Library Compilation & Linking
### Ⅰ `add_library`: Create Libraries

```cmake

# Basic usage
add_library(pipeline STATIC pipeline.cpp)

# Customize output name
set_target_properties(pipeline PROPERTIES OUTPUT_NAME "pipeline")

# Custom output paths
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib) # Shared libs

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/static) # Static libs
```

### Ⅱ `target_link_libraries`: Link Libraries

```cmake

target_link_libraries(main PUBLIC pipeline) # Links 'pipeline' to 'main'
```

### Ⅲ `find_package`: Link Prebuilt Libraries (with `.cmake` Files)

```cmake

find_package(OpenCV REQUIRED) # Finds OpenCV

target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(main PUBLIC ${OpenCV_LIBRARIES})
```

### Ⅳ `target_link_directories`: Link Prebuilt Libraries (without `.cmake` Files)

```cmake

target_link_directories(main PRIVATE /opt/opencv/lib)

target_link_libraries(main PUBLIC libcore.lib)

# Link all .lib files in a directory
file(GLOB WIN_LIBS "${CMAKE_SOURCE_DIR}/lib/*.lib")

target_link_libraries(main PRIVATE ${WIN_LIBS})
```

## 2. `target_include_directories`: Add Header Search Paths
1. Specifies header search paths for a target.  
2. `PUBLIC`: Path visible to the target and its dependents.  
   `PRIVATE`: Path visible only to the target.

```cmake

target_include_directories(main PUBLIC include)
```
**Note**: Only affects `#include` paths, not library linking.  

## 3. `set_target_properties`: Configure Target Properties

```cmake

# Rename output
set_target_properties(hello_static PROPERTIES OUTPUT_NAME "hello")

# Set version
set_target_properties(hello PROPERTIES VERSION 1.2 SOVERSION 1)

# Custom output paths
set_target_properties(mul PROPERTIES

LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib

RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin

)

# Debug suffix
set_target_properties(hello PROPERTIES DEBUG_POSTFIX _d)

```

## 4. `list`: Manipulate Lists

```cmake

# Add elements
list(APPEND my_list "item1" "item2")

list(PREPEND my_list "item0")

# Remove elements
list(REMOVE_ITEM my_list "item0.5")

list(REMOVE_AT my_list 0)

# Transform elements
list(TRANSFORM SOURCES PREPEND "src/") # Adds "src/" prefix

list(TRANSFORM PATHS REPLACE "/old/" "/new/") # Replaces paths

# Utilities
list(LENGTH my_list list_length) # Get length

list(JOIN my_list ";" joined_str) # Join into a string

```