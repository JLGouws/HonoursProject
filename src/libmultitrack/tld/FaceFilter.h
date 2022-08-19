/*
 * VarianceFilter.h
 *
 *  Created on: Aug 18, 2022
 *      Author: J L Gouws
 */

#ifndef FACEFILTER_H_
#define FACEFILTER_H_

#include <opencv/cv.h>
#include <opencv2/objdetect.hpp> 

#include "IntegralImage.h"
#include "DetectionResult.h"

namespace tld
{

class FaceFilter
{
  public:
    bool enabled;

    DetectionResult *detectionResult;

    FaceFilter(long frame);
    virtual ~FaceFilter();

    float minOverlap;
    int *windowOffsets;

    void nextIteration(const cv::Mat &img, long frame);
    bool filter(int idx);
  private:
    std::vector<cv::Rect> faces;
    long frameNumber;
    cv::CascadeClassifier *detector;
    float calcFace(int *off);
    float scoreThreshold;
    float nmsThreshold;
    int topK;
};

} /* namespace tld */
#endif /* FACEFILTER_H_ */
