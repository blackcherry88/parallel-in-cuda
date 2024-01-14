cmake_minimum_required(VERSION 3.17)
project(vec_add CUDA CXX C)  # enable cuda language

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # enable cuda-gdb
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
endif()

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cu)
# target_sources(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}.cu)

# you can also set as all-major
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)