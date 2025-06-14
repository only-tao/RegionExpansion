cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(density LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH "~/.conda/envs/pdflow/lib/python3.8/site-packages/torch/share/cmake/Torch")
# set(CUDA_TOOLKIT_ROOT "/usr/local/cuda")
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
message(STATUS "CUDA Toolkit found at: ${CUDA_TOOLKIT_ROOT_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ")
message(STATUS "cxx flags: ${CMAKE_CXX_FLAGS}") #-- cxx flags:  -D_GLIBCXX_USE_CXX11_ABI=1
message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
# -- TORCH_CXX_FLAGS: -D_GLIBCXX_USE_CXX11_ABI=1
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
# message(STATUS "cxx flags: ${CMAKE_CXX_FLAGS}")

message(STATUS "torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "cuda include dirs: ${CUDA_INCLUDE_DIRS}")
include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

add_executable(density_exec density.cu)
add_library(density SHARED density.cu)
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")


target_link_libraries(density_exec
  ${CUDA_LIBRARIES}
  ${TORCH_LIBRARIES}
)
target_link_libraries(density
  ${CUDA_LIBRARIES}
  ${TORCH_LIBRARIES}
)
set_property(TARGET density_exec PROPERTY CUDA_STANDARD 17)
set_property(TARGET density PROPERTY CUDA_STANDARD 17)
set_property(TARGET density_exec PROPERTY CXX_STANDARD 17)
set_property(TARGET density PROPERTY CXX_STANDARD 17)
set_property(TARGET density_exec PROPERTY CUDA_ARCHITECTURES 70) 
set_property(TARGET density PROPERTY CUDA_ARCHITECTURES 70) 
# set_output_dir(density_exec)
# add_custom_command(TARGET density POST_BUILD
#   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:density> ${CMAKE_SOURCE_DIR}
# )
