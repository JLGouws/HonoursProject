/*
 * FaceFilter.cpp
 *
 *  Created on: Aug 18, 2022
 *      Author: J L Gouws
 */

#include "FaceFilter.h"
#include <opencv2/dnn.hpp> 
#include <opencv2/objdetect.hpp> 

#include "IntegralImage.h"
#include "DetectorCascade.h"

using namespace cv;

namespace tld
{

FaceFilter::FaceFilter(long frame = 0)
{
    enabled = true;
    frameNumber = frame;
    detector = NULL;
}

FaceFilter::~FaceFilter()
{
  detector->release();
  detector = NULL
}

void FaceFilter::nextIteration(const Mat &img, long frame)
{
    if(frame != frameNumber)
    {
        if(detector == NULL;
          detector = FaceDetectorYN::create("face_detection_yunet_2022mar.onnx", "", img.size(), scoreThreshold, nmsThreshold, topK);

        if(!enabled) return;

        detector->detect(img, faces);
        frameNumber++;
    }
}

bool FaceFilter::filter(int i)
{
    if(!enabled) return true;
/*
    float bboxvar = calcFace(windowOffsets + TLD_WINDOW_OFFSET_SIZE * i);

    detectionResult->variances[i] = bboxvar;

    if(bboxvar < minVar)
    {
        return false;
    }
    */

    return true;
}

} /* namespace tld */
