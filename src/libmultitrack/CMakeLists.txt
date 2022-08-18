
include_directories(imacq
    mftracker
    tld
    ../3rdparty/cvblobs
    ../libkcf
    ${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

add_library(libmultitrack
    imacq/ImAcq.cpp
    mftracker/BB.cpp
    mftracker/BBPredict.cpp
    mftracker/FBTrack.cpp
    mftracker/Lk.cpp
    mftracker/Median.cpp
    tld/Clustering.cpp
    tld/DetectionResult.cpp
    tld/DetectorCascade.cpp
    tld/EnsembleClassifier.cpp
    tld/ForegroundDetector.cpp
    tld/MedianFlowTracker.cpp
    tld/NNClassifier.cpp
    tld/MultiTrack.cpp
    tld/TLDUtil.cpp
    tld/VarianceFilter.cpp
    tld/FaceFilter.cpp
    imacq/ImAcq.h
    mftracker/BB.h
    mftracker/BBPredict.h
    mftracker/FBTrack.h
    mftracker/Lk.h
    mftracker/Median.h
    tld/Clustering.h
    tld/DetectionResult.h
    tld/DetectorCascade.h
    tld/EnsembleClassifier.h
    tld/ForegroundDetector.h
    tld/IntegralImage.h
    tld/MedianFlowTracker.h
    tld/NNClassifier.h
    tld/NormalizedPatch.h
    tld/MultiTrack.h
    tld/TLDUtil.h
    tld/VarianceFilter.h
    tld/FaceFilter.h
    )

target_link_libraries(libmultitrack ${OpenCV_LIBS} kcf)

set_target_properties(libmultitrack PROPERTIES OUTPUT_NAME multitrack)

