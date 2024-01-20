cmake_minimum_required(VERSION 3.17)
project(paraell-cuda)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(lib)
add_subdirectory(hellocuda)
add_subdirectory(device_info)
add_subdirectory(vec_add)
add_subdirectory(vec_add2)
add_subdirectory(tile_matrixmult)