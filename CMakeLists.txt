cmake_minimum_required(VERSION 2.8)
project(test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
message(STATUS "C++11 support has been enabled by default.")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")

find_package(OpenCV REQUIRED)

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

include_directories(src) 
include_directories (${PROJECT_SOURCE_DIR}/include	${OpenCV_INCLUDE_DIRS})

FILE(GLOB_RECURSE sourcefiles "src/*.cpp")

include_directories(${OpenCV_INCLUDE_DIRS} build)
link_directories(${PROJECT_BINARY_DIR})
list(APPEND fa_required_libs ${OpenCV_LIBS} seeta_facedet_lib seeta_fa_lib)

add_executable( KCF ${sourcefiles} )
target_link_libraries( KCF ${OpenCV_LIBS} ${fa_required_libs})




