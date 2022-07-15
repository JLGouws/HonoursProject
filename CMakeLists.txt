cmake_minimum_required(VERSION 2.8)

project(tld_tracker)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -O3 -Wno-long-long -fno-omit-frame-pointer")

FIND_PACKAGE( OpenCV REQUIRED )
link_directories ( ${OpenCV_LIB_DIR} )
MESSAGE(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR}")

IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
ENDIF ()

include_directories ( ${OpenCV_INCLUDE_DIRS} )
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES( ${CMAKE_BINARY_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/tldSrc/ )
include_directories(${CMAKE_SOURCE_DIR}/tracking/ )
include_directories(${CMAKE_SOURCE_DIR}/kcfSrc/ ${CMAKE_SOURCE_DIR}/kcfSrc/piotr_fhog ${CMAKE_SOURCE_DIR}/kcfSrc/cn)

add_subdirectory(tldSrc)
add_subdirectory(kcfSrc)
add_subdirectory(tracking)

add_executable(predator runTLD.cpp videoHandler.hpp)
target_link_libraries(predator ${OpenCV_LIBS} tld kcf track_utils)

add_executable(cameraDemo cameraDemo.cpp)
target_link_libraries(cameraDemo ${OpenCV_LIBS} tld kcf track_utils)

add_executable(vidDemo vidDemo.cpp)
target_link_libraries(vidDemo ${OpenCV_LIBS} tld kcf track_utils)
