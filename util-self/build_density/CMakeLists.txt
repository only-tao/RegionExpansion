cmake_minimum_required(VERSION 3.10)

project(ExampleProject VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
# find_package(CUDAToolkit REQUIRED)
#cpp abi to 0
include_directories(${Python3_INCLUDE_DIRS})
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
# include_directories(/home/dengt35/build/include)
include_directories(${TORCH_INCLUDE_DIRS})

link_directories(/path_to_you/util-self/build) # change this !!!! 
link_directories(/path_to_you/libtorch/lib) # change this !!!! this is what you install just.

add_library(example MODULE example.cpp)
target_compile_options(example PRIVATE -O3 -Wall -fPIC)

smessage(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
target_link_libraries(example PRIVATE  density ${Python3_LIBRARIES} ${TORCH_LIBRARIES} torch_python )

execute_process(
    COMMAND python3-config --extension-suffix
    OUTPUT_VARIABLE Python3_EXTENSION_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# python3-config --extension-suffix
set_target_properties(example PROPERTIES PREFIX "" SUFFIX ${Python3_EXTENSION_SUFFIX})
add_custom_command(TARGET example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:example> ${CMAKE_SOURCE_DIR}
)