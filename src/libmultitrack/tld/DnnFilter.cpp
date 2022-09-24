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
#include <cstdlib>                                                              
#include <cmath>  
#include <iostream>  

using namespace cv;

namespace tld
{

DnnFilter::DnnFilter(long frame = 0)
{
    enabled = true;
    frameNumber = frame;
    minOverlap = 0.75;
    minConfidence = 0.2;
    windows = NULL;
    sFaces = NULL;
    numWindows = -1;
    init = false;
}

DnnFilter::~DnnFilter()
{
  init = false;
}

void DnnFilter::calcFaces()
{
    if(sFaces == NULL)
    {
      sFaces = new bool[numWindows];
      rPatch = new bool[numWindows];
    }

    #pragma omp parallel for

    for(int i = 0; i < numWindows; i++)
    {

      rPatch[i] = false;
      int *off = windows + TLD_WINDOW_SIZE * i;

      float max = 0.,
            overlap;

      for (const auto &face :faces) 
      {
        overlap = tldOverlapBBRect(off, face);
        max = overlap > max ? overlap : max;
      }

      sFaces[i] = max > minOverlap;
//      if (!sFaces[i])
//        rPatch[i] = rand() < RAND_MAX / 100;

    }
}

float DnnFilter::calcFace(int *off)
{
    float max = 0.,
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
        if(!init)
        {
          srand(0);
          detector = dnn::readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel");
          init = true;
        }

        if(!enabled) return;

        imw = img.cols;
        imh = img.rows;
        
        faces.clear();

        Mat blob = dnn::blobFromImage(img, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false, false);

        detector.setInput(blob, "data");

        Mat results = detector.forward("detection_out");

        Mat detectionMat(results.size[2], results.size[3], CV_32F, results.ptr<float>());

        for(int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);
            if(confidence > minConfidence)
            {
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * imw);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * imh);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * imw);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * imh);

                Rect object((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));

                faces.push_back(object);
            }
        }

        calcFaces();

        frameNumber = frame;
    }
}

bool DnnFilter::filter(int i)
{

    return sFaces[i] || rPatch[i];
}

bool DnnFilter::filter(int i, const Rect *trackerBB)
{
  if (trackerBB == NULL)
    return sFaces[i]; 
  else
    return sFaces[i] || rPatch[i];
}

bool DnnFilter::filter(int i, const bool valid)
{
    return (valid && rPatch[i]) || sFaces[i];
}

} /* namespace tld */
