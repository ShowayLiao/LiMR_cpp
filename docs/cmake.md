<div align="center">

<h1>cmake配置指南</h1>

</div>

# 📌 cpp代码生成全流程
## 预处理
1. 预处理是 C/C++ 编译过程的第一步，它主要负责源代码中的宏替换、文件包含、条件编译等操作
2. `#define`：定义宏。
3. `#include`：包含头文件。
4. `#ifdef`、`#ifndef`、`#endif`：条件编译。
5. `#pragma`：给编译器传递特殊指令（通常用于平台特定的优化）。
## 编译
1. 编译阶段的主要任务是将经过预处理的源代码转换为汇编代码，生成文件(.s)
## 汇编
1. 编译器生成的汇编代码转化为机器代码，转换为.o文件，但这些文件依旧不是可执行文件（缺少引用的库链接）
## 链接
1. 将编译生成的目标文件组合成最终的可执行文件（或库文件）的过程。链接有两种形式：静态链接和动态链接。
### 静态链接
1. 静态链接发生在编译时，它将所有的目标文件和库文件中的代码和数据结合在一起，生成一个独立的可执行文件。也就是说，使用`add_library(pipeline STATIC pipeline.cpp)`生成`pipeline.lib`文件后，在链接阶段，这个文件会被嵌入`main.exe`中。在多个程序共用一个库时，这种方法会在每一个程序中嵌入这个库，会造成冗余。
### 动态链接
1. 动态链接在运行时进行，它将程序和共享库（例如 .so 文件或 .dll 文件）在运行时链接起来。

<details style="color:rgb(128,128,128)">
<summary>库的区别</summary>

 
#### 静态库STATIC
1. 编译为 .a（Linux）或 .lib（Windows）文件。
2. 链接时被完整复制到可执行文件中，导致文件体积增大，但运行时无需依赖外部库。
```
add_library(mylib STATIC src/file1.cpp src/file2.cpp)
```
#### 共享库SHARED
1. 编译为 .so（Linux）或 .dll（Windows）文件。
2. 运行时动态加载，多个程序可共享同一个库，减少内存占用。
```
add_library(mylib SHARED src/file1.cpp src/file2.cpp)
```
#### 模块库MODULE
1. 类似共享库，但不直接链接到可执行文件，而是通过 dlopen()（Linux）或 LoadLibrary()（Windows）动态加载。
```
add_library(mymodule MODULE src/module.cpp)
```
#### 目标文件集合（OBJECT）
1. 不生成库文件，仅编译源文件为 .o/.obj 文件。
2. 可被多个目标重复使用，减少重复编译。
```
add_library(myobjects OBJECT src/file1.cpp src/file2.cpp)
add_executable(myapp $<TARGET_OBJECTS:myobjects> src/main.cpp)
```
</details>

# 📌 动态库文件
1. 动态库文件生成时，存在lib（mingw生成为dll.a）和dll两个文件（windows），前者是导入文件，仅有函数定义；后者是动态库文件，包含完整的函数。
2. 基本上所有的三方库都是动态库。
## 动态库文件使用
### 隐式调用
1. 隐式链接调用就是在程序开始执行时就将dll文件加载到程序当中且整个执行过程无法分离。调用十分简单，但需要h,lib和dll文件的支持。
2. 使用：
   1. 包含对应的头文件（取决于dll中头文件是如何定义的）
   2. 链接导入库（lib文件而非dll）
   3. 运行时需要将dll放在程序目录，或者是设置环境变量path
### 显式调用
1. 使用系统api加载库： `LoadLibrary("MathLib.dll");`
2. 获取函数指针（`GetProcAddress/dlsym`）
3. 调用函数后卸载库（`FreeLibrary/dlclose`）。
```cpp
 #include <windows.h>
 #include <iostream>

 typedef int (*AddFunc)(int, int);

 int main() {
     HINSTANCE hDll = LoadLibrary("MathLib.dll");
     if (hDll) {
         AddFunc add = (AddFunc)GetProcAddress(hDll, "add");
         if (add) {
             std::cout << add(2, 3) << std::endl; // 输出5
         }
         FreeLibrary(hDll);
     }
     return 0;
 }

```

# 📌 cmake常见变量
## 全局变量[ref](https://blog.csdn.net/Akutamatsu/article/details/132720660)
1. `${CMAKE_SOURCE_DIR}`：这个变量表示项目的根源目录，即包含 `CMakeLists.txt` 文件的目录。通常用于指定源代码文件的位置。
2. `${CMAKE_BINARY_DIR}`：表示 CMake 生成的构建文件(`./build`)（例如编译器生成的中间文件和可执行文件）的根目录。这通常用于指定构建文件的输出位置。
3. `${CMAKE_CURRENT_SOURCE_DIR}`：表示当前处理的 `CMakeLists.txt` 文件所在的目录。这个变量在多目录项目中很有用。
4. `${CMAKE_CURRENT_BINARY_DIR}`：表示当前处理的 `CMakeLists.txt` 文件的构建目录，通常与`${CMAKE_CURRENT_SOURCE_DIR}`相对应。
5. `${CMAKE_MODULE_PATH}`：一个包含了额外的模块查找路径的变量。这可以用于自定义模块的位置。
6. `${CMAKE_INSTALL_PREFIX}`：指定安装目录的根路径。在安装项目时，可用于确定安装文件的位置。
7. `${CMAKE_LIBRARY_OUTPUT_DIRECTORY}` 和` ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}`：分别指定库文件和可执行文件的输出目录。
8. `${CMAKE_BUILD_TYPE}`：指定构建的类型，如 Debug、Release 等。这在控制编译选项和优化时非常有用。
9. `EXECUTABLE_OUTPUT_PATH`： 指定最终的可执行文件的位置
10. `LIBRARY_OUTPUT_PATH`： 设置库文件的输出目录
## 自定义变量方法[ref](https://blog.csdn.net/weixin_42036710/article/details/147741035)
1. 隐式定义: 就是 PROJECT 指令,他会隐式的定义<projectname>_BINARY_DIR 和<projectname>_SOURCE_DIR 两个变量。
2. 显式定义: 使用 SET 指令,就可以构建一个自定义变量了。`SET(HELLO_SRC main.c)`,就可以通过${HELLO_SRC}来引用这个自定义变量了
## 编译器变量[ref](https://blog.csdn.net/Long_xu/article/details/147074866)
1. `find_program`:[ref](https://blog.csdn.net/fengbingchun/article/details/127338012)
2. `CMAKE_CXX_COMPILER`C++ 语言编译器。指定用于编译 C++ 语言源代码的编译器可执行文件的完整路径。与 CMAKE_C_COMPILER 类似，如果未显式设置，CMake 会自动搜索 C++ 编译器。
3. `CMAKE_BUILD_TYPE`: 构建类型。指定构建类型，影响编译器的优化级别和调试信息的生成。用户可以通过在命令行中指定 `-DCMAKE_BUILD_TYPE=<build_type>` 来覆盖此变量的值。常用值:

# 📌 cmake基础指令详解
## 1. cmake：库的编译、链接
### Ⅰ 库的编译：`add_library`创建动态库、静态库[ref](https://blog.csdn.net/weixin_43510208/article/details/148629726)
1. 基本用法
```cpp
add_library(<target_name> [STATIC | SHARED | MODULE] [EXCLUDE_FROM_ALL] [source1] [source2 ...])
add_library(pipeline STATIC pipeline.cpp)// 用源码pipeline.cpp生成静态STATIC对象 pipeline.lib
```
2. 支持设置属性
```cpp
add_library(pipeline STATIC pipeline.cpp)
SET_TARGET_PROPERTIES (pipeline PROPERTIES OUTPUT_NAME "pipeline")// 修改输出名字
```
3. 改变输出路径
```cpp
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)        # 动态库
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/static)     # 静态库
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)        # 执行文件
```
4. 注意⚠️：对于外部依赖库，属于预编译对象，不需要通过编译
### Ⅱ 自定义库的链接：`target_link_libraries`链接库[ref](https://blog.csdn.net/mataojie/article/details/133352276)
1. 基本用法
```cpp
target_link_libraries(<target> ... <item>... ...)// 这里的item可以是导入库，也可以是通过library生成的文件
target_link_libraries(main PUBLIC pipeline)
```
### Ⅲ 三方预编译库的链接（有`.cmake`文件）：`find_package`[ref](https://blog.csdn.net/m0_73640344/article/details/144246891)
1. 基本用法
```cpp
find_package(<Package> [version] [REQUIRED] [COMPONENTS components...])
//<Package>：要查找的包名称。
//REQUIRED：表示该包为必需，如果未找到则停止配置过程并报错。
find_package(OpenCV REQUIRED)// 查找opencv
// 查找成功会会设置变量：<Package>_FOUND、<Package>_INCLUDE_DIRS、<Package>_LIBRARIES
target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS})// 将找到的include路径包含进main的头文件搜索范围
target_link_libraries(main PUBLIC ${OpenCV_LIBRARIES})// 在链接阶段，将找到的lib路径下的所有lib连接到main
```
2. `find_package`的找包方法
   1. module模式（优先）：查找`CMAKE_MODULE_PATH`变量的路径（路径下有`Find<Package>.cmake`）或者是 CMake 安装目录下的 `Modules/`
   2. config模式：`<Package>_DIR`变量指定路径（路径下有`<Package>Config.cmake`）或者`CMAKE_PREFIX_PATH`变量指定或者PATH环境变量中的路径
```cpp
set(OpenCV_DIR "/opt/opencv4/lib/cmake/opencv4")
set(CMAKE_PREFIX_PATH "/opt/boost;/opt/opencv4")//多个路径用；隔开
```
### Ⅳ 三方预编译库的链接（没有`.cmake`文件）：target_link_directories
1. 基本用法：该函数用于在链接时，为目标增加搜索的链接路径，便于`target_link_libraries`寻找库文件
```cpp
target_link_directories(main PRIVATE /opt/opencv/lib) 
target_link_libraries(main PUBLIC libcore.lib)
```
2. 也可以指定路径的全局变量，通过循环将所有`lib`文件链接到`main`上
```cpp
file(GLOB WIN_LIBS "${CMAKE_SOURCE_DIR}/lib/*.lib") // 寻找所有lib结尾的文件
target_link_libraries(main PRIVATE ${WIN_LIBS})// 将所有文件链接到main上
```


## 2. cmake:target_include_directories [ref](https://blog.csdn.net/FL1768317420/article/details/137130614)
1. 这个命令是为指定编译文件(add_executable()或add_library()创建的)增加头文件的搜索范围。
2. `include_directories`会对当前CMakeLists.txt文件的目标文件生效，并会通过`add_subdirectory`调用传递到子目录；`target_include_directories`则针对指定的目标文件生效
```cpp
target_include_directories(<target>
    [SYSTEM] [BEFORE]
    <INTERFACE|PUBLIC|PRIVATE> [path1 [path2 ...]]
    [<INTERFACE|PUBLIC|PRIVATE> [path3 ...] ...]
)// public:这个新增的搜索范围对依赖于target的程序依然可见
// PRIVATE：对依赖者不可见
```
3. 注意⚠️：**此命令​​不处理库文件的链接​​，仅解决 #include 语句的路径问题**。


## 3. cmake：SET_TARGET_PROPERTIES设置常见内置属性 [ref](https://blog.csdn.net/challenglistic/article/details/130130789)
1. 改变输出名字`SET_TARGET_PROPERTIES (hello_static PROPERTIES OUTPUT_NAME "hello")`
2. 设置版本号`SET_TARGET_PROPERTIES (hello PROPERTIES VERSION 1.2 SOVERSION 1)`
3. 单独设置输出路径
```cpp
set_target_properties(mul 
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY 
    ${PROJECT_SOURCE_DIR}/build/lib
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/build/bin
    ARCHIVE_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/build/static
)
```
4. 指定Debug模式下目标文件名的后缀`SET_TARGET_PROPERTIES (hello PROPERTIES DEBUG_POSTFIX _d)`
5. 自定义属性
```cpp
add_library(test_lib SHARED ${ALL_SRCS})
# 为目标文件 test_lib 创造一个 _STATUS_ 属性，属性值为 shared
set_target_properties(test_lib PROPERTIES _STATUS_ shared)
 
# 获取 test_lib 的 _STATUS_ 属性并保存到 var
get_target_property(var test_lib _STATUS_)
message("===============================")
message("= test_lib的_STATUS_属性: ${var}")
message("===============================")

```

## 4. cmake：list 操作文件列表[ref](https://blog.csdn.net/haokan123456789/article/details/148355825)
1. list 命令支持多种子命令，用于实现列表的创建、修改、查询和处理等操作。
2. APPEND：向列表末尾添加元素
```cpp
list(APPEND my_list "item1" "item2")  # 添加多个元素
# 结果：my_list = ["item1", "item2"]
```
3. PREPEND：向列表开头添加元素
```cpp
list(PREPEND my_list "item0")
# 结果：my_list = ["item0", "item1", "item2"]
```
4. INSERT：在指定索引处插入元素
```cpp
list(INSERT my_list 1 "item0.5")  # 在索引 1 处插入元素
# 结果：my_list = ["item0", "item0.5", "item1", "item2"]
```
5. REMOVE_ITEM：删除列表中指定值的元素
```cpp
list(REMOVE_ITEM my_list "item0.5")  # 删除值为 "item0.5" 的元素
# 结果：my_list = ["item0", "item1", "item2"]
```
6. REMOVE_AT：删除指定索引处的元素
```cpp
list(REMOVE_AT my_list 0)  # 删除索引 0 处的元素
# 结果：my_list = ["item1", "item2"]
```
7. REMOVE_DUPLICATES：移除列表中的重复元素
```cpp
list(APPEND my_list "item1")  # 添加重复元素
# 此时 my_list = ["item1", "item2", "item1"]
list(REMOVE_DUPLICATES my_list)
# 结果：my_list = ["item1", "item2"]
```
8. LENGTH：获取列表长度（元素个数）
```cpp
list(LENGTH my_list list_length)
message(STATUS "列表长度：${list_length}")  # 输出：列表长度：2
```
9. GET：获取列表中指定索引的元素
```cpp
list(GET my_list 0 first_item)  # 获取索引 0 处的元素
message(STATUS "第一个元素：${first_item}")  # 输出：第一个元素：item1
```
10. JOIN：将列表元素用指定分隔符连接成字符串
```cpp
list(JOIN my_list ";" joined_string)
message(STATUS "连接后的字符串：${joined_string}")  # 输出：item1;item2
```
11. REVERSE：反转列表顺序
```cpp
list(REVERSE my_list)
# 结果：my_list = ["item2", "item1"]
```
12. SORT：对列表进行排序（按字母顺序）
```cpp
list(APPEND my_list "item3" "item0")
# 此时 my_list = ["item2", "item1", "item3", "item0"]
list(SORT my_list)
# 结果：my_list = ["item0", "item1", "item2", "item3"]
```
13. TRANSFORM: 列表元素变换(详见[ref](https://blog.csdn.net/haokan123456789/article/details/148355825))
```cpp
set(SOURCES "main.cpp" "utils.cpp")
 
# 为每个源文件添加 "src/" 前缀（等效于 "src/main.cpp" "src/utils.cpp"）
list(TRANSFORM SOURCES PREPEND "src/")
 
# 为每个源文件添加 "_test" 后缀（等效于 "main_test.cpp" "utils_test.cpp"）
list(TRANSFORM SOURCES APPEND "_test.cpp")


set(PATHS "/old/path/file1" "/old/path/file2")
 
# 将 "/old/path/" 替换为 "/new/path/"
list(TRANSFORM PATHS REPLACE "/old/path/" "/new/path/")
```
---


