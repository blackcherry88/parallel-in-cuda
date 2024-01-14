cmake_minimum_required(VERSION 3.17)
project(wb LANGUAGES CUDA CXX C VERSION 1.0.1)  # enable cuda language
set(CMAKE_CUDA_STANDARD 11)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # enable cuda-gdb
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
endif()

file(GLOB  wbhdr ./*.hpp ./*.h )
file(GLOB  wbsrc ./*.cpp ./*.c )

include(GNUInstallDirs)
add_library(${PROJECT_NAME} SHARED ${wbsrc})

set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER ${PROJECT_NAME}/${wbhdr})

# configure_file(${PROJECT_NAME}.pc.in ${PROJECT_NAME}.pc @ONLY)
target_include_directories(${PROJECT_NAME} PRIVATE .)

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc
#     DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

