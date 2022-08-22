/*
 * DnnFilter.cpp
 *
 *  Created on: Aug 18, 2022
 *      Author: J L Gouws
 */

#include "DnnFilter.h"

#include "IntegralImage.h"
#include "TLDUtil.h"
#include "DetectorCascade.h"

using namespace cv;

namespace tld
{

DnnFilter::DnnFilter(long frame = 0)
{
    enabled = true;
    frameNumber = frame;
    minOverlap = 0.6;
    windowOffsets = NULL;
    init = false;
}

DnnFilter::~DnnFilter()
{
  init = false;
}

float DnnFilter::calcFace(int *off)
{
    float max = 0,
          overlap;
    for (const auto &face :faces) 
    {
      overlap = tldOverlapBBRect(off, face);
      max = overlap > max ? overlap : max;
    }
    return max;
}

void DnnFilter::nextIteration(const Mat &img, long frame)
{
    if(frame != frameNumber)
    {
        if(init)
        {
          detector = dnn::readNetFromCaffe("deploy.prototxt", "models/res10_300x300_ssd_iter_140000_fp16.caffemodel");
          init = true;
        }

        if(!enabled) return;

        float *res;
        imw = img.cols;
        imh = img.rows;
        
        faces.clear();

        Mat blob = dnn::blobFromImage(img, 1.0, Size(300, 300), Scalar(104., 117., 123.), false, false);

        detector.setInput(blob);

        Mat results = detector.forward();

        for (int i = 0; i < results.size[2]; i++) 
        {
          res = results.at<float *>(0, 0, i);
          if (res[2] > minConfidence)
            faces.push_back(Rect(Point((int) imw * res[3], (int) imh * res[4]), Point((int) imw * res[5], (int) imh * res[6])));
        }

        frameNumber++;
    }
}

bool DnnFilter::filter(int i)
{
    if(!enabled) return true;

    float bboxoverlap = calcFace(windowOffsets + TLD_WINDOW_OFFSET_SIZE * i);

    detectionResult->overlaps[i] = bboxoverlap;

    if(bboxoverlap < minOverlap)
    {
        return false;
    }

    return true;
}

} /* namespace tld */
