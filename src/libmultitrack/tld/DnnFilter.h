/*
 * VarianceFilter.h
 *
 *  Created on: Aug 18, 2022
 *      Author: J L Gouws
 */

#ifndef DNNFILTER_H_
#define DNNFILTER_H_

#include <opencv/cv.h>
#include <opencv2/objdetect.hpp> 
#include <opencv2/dnn/dnn.hpp> 

#include "IntegralImage.h"
#include "DetectionResult.h"

namespace tld
{

class DnnFilter
{
  public:
    bool enabled;

    DetectionResult *detectionResult;

    DnnFilter(long frame);
    virtual ~DnnFilter();

    float minOverlap;
    int *windows;
    int numWindows;
    bool *sFaces;
    bool *rPatch;

    void nextIteration(const cv::Mat &img, long frame);
    bool filter(int idx);
    bool filter(int idx, const cv::Rect *trackerBB);
    bool filter(int idx, const bool valid);
    float minConfidence;
  private:
    std::vector<cv::Rect> faces;
    long frameNumber;
    cv::dnn::Net detector;
    float calcFace(int *off);
    void calcFaces();
    int imw, imh;
    bool init;
};

} /* namespace tld */
#endif /* FACEFILTER_H_ */
