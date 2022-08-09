
link_directories(${OpenCV_LIB_DIR})

include_directories(main
    ../libopentld/imacq
    ../libopentld/mftracker
    ../libopentld/tld
    ../3rdparty/cvblobs
    ../libkcf
    ${OpenCV_INCLUDE_DIRS})

if(NOT USE_SYSTEM_LIBS)
    include_directories(../3rdparty/libconfig)
endif(NOT USE_SYSTEM_LIBS)

#-------------------------------------------------------------------------------
# opentld
add_executable(vidDemo
    vidDemo.cpp)

add_executable(OTBtest
    runTLD.cpp
    videoHandler.hpp)

target_link_libraries(vidDemo main libopentld cvblobs config++ kcf ${OpenCV_LIBS})

target_link_libraries(OTBtest main libopentld cvblobs config++ kcf ${OpenCV_LIBS})

install(TARGETS vidDemo DESTINATION bin)

#--------------------------------------------------------------------------------
