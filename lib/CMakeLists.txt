cmake_minimum_required(VERSION 3.17)
project(wb LANGUAGES CUDA CXX C VERSION 1.0.1)  # enable cuda language

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # enable cuda-gdb
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
endif()

file(GLOB  wbsrc ./*.cpp ./*.c )

add_library(${PROJECT_NAME} SHARED ${wbsrc})

set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})

target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
