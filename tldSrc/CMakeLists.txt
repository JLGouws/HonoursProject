cmake_minimum_required(VERSION 2.8)

set(TLD_LIB_SRC tldTracker.cpp tldTracker.hpp tldDetector.cpp tldDetector.hpp tldEnsembleClassifier.cpp tldEnsembleClassifier.hpp tldModel.cpp tldModel.hpp tldUtils.cpp tldUtils.hpp trackerMedianFlow.cpp)

add_library(tld STATIC ${TLD_LIB_SRC})
target_link_libraries(tld ${OpenCV_LIBS})
set_target_properties(tld PROPERTIES VERSION 1.0.0 SOVERSION 1)
