cmake_minimum_required(VERSION 2.8)

set(TRACK_UTILS_SRC tracking_utils.cpp tracking_utils.hpp)

add_library(track_utils STATIC ${TRACK_UTILS_SRC})
target_link_libraries(track_utils ${OpenCV_LIBS})
set_target_properties(track_utils PROPERTIES VERSION 1.0.0 SOVERSION 1)
