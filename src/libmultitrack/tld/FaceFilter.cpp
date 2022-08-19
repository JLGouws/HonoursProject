/*
 * FaceFilter.cpp
 *
 *  Created on: Aug 18, 2022
 *      Author: J L Gouws
 */

#include "FaceFilter.h"

#include "IntegralImage.h"
#include "TLDUtil.h"
#include "DetectorCascade.h"

using namespace cv;

namespace tld
{

FaceFilter::FaceFilter(long frame = 0)
{
    enabled = true;
    frameNumber = frame;
    minOverlap = 0.6;
    detector = NULL;
    windowOffsets = NULL;
}

FaceFilter::~FaceFilter()
{
  delete detector;
  detector = NULL;
}

float FaceFilter::calcFace(int *off)
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

void FaceFilter::nextIteration(const Mat &img, long frame)
{
    if(frame != frameNumber)
    {
        if(detector == NULL)
          detector = new CascadeClassifier("haarcascade_frontalcatface_extended.xml");

        if(!enabled) return;

        detector->detectMultiScale(img, faces);
        frameNumber++;
    }
}

bool FaceFilter::filter(int i)
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
