/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
* OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/

/*
 * Detector.cpp
 *
 *  Created on: Nov 16, 2011
 *      Author: Georg Nebehay
 */

#include "Detector.h"

#include <algorithm>

#include <iostream>

#include "TLDUtil.h"

using namespace cv;
using namespace std;

namespace tld
{

//TODO: Convert this to a function
#define sub2idx(x,y,imgWidthStep) ((int) (floor((x)+0.5) + floor((y)+0.5)*(imgWidthStep)))

Detector::Detector() : Detector((long) 0)
{
  /*
    objWidth = -1; //MUST be set before calling init
    objHeight = -1; //MUST be set before calling init
    useShift = 1;
    imgHeight = -1;
    imgWidth = -1;

    shift = 0.1;
    minScale = -10;
    maxScale = 10;
    minSize = 25;
    imgWidthStep = -1;

    numTrees = 10;
    numFeatures = 13;

    initialised = false;

    frameNumber = 0;

    foregroundDetector = new ForegroundDetector();
    dnnFilter = new DnnFilter();
    ensembleClassifier = new EnsembleClassifier();
    nnClassifier = new NNClassifier();
    clustering = new Clustering();

    detectionResult = new DetectionResult();
    */
    //this((long) 0);
}

Detector::Detector(long frame) : Detector(new DnnFilter(frame - 1), frame)
{
}

Detector::Detector(DnnFilter *varFil) : Detector(varFil, long(0))
{
}

Detector::Detector(DnnFilter *dnnFil, long frame)
{
    objWidth = -1; //MUST be set before calling init
    objHeight = -1; //MUST be set before calling init
    useShift = 1;
    imgHeight = -1;
    imgWidth = -1;

    shift = 0.1;
    minScale = -10;
    maxScale = 10;
    minSize = 25;
    imgWidthStep = -1;

    initialised = false;

    frameNumber = frame;

    dnnFilter = dnnFil;
    cfClassifier = new CFClassifier();
    detectorBB = NULL;
}

Detector::~Detector()
{
    release();

    if(dnnFilter)
    {
      delete dnnFilter;
      dnnFilter = NULL;
    }
}

void Detector::init(int numWindows0, int *windows0, int *windowOffsets0, int numScales0, Size *scales0)
{
    if(imgWidth == -1 || imgHeight == -1 || imgWidthStep == -1 || objWidth == -1 || objHeight == -1)
    {
        //printf("Error: Window dimensions not set\n"); //TODO: Convert this to exception
    }

    initialised = true;
}

void Detector::init()
{
    if(imgWidth == -1 || imgHeight == -1 || imgWidthStep == -1 || objWidth == -1 || objHeight == -1)
    {
        //printf("Error: Window dimensions not set\n"); //TODO: Convert this to exception
    }

    initialised = true;
}

void Detector::release()
{
    if(!initialised)
    {
        return; //Do nothing
    }

    initialised = false;


    objWidth = -1;
    objHeight = -1;

}

void Detector::cleanPreviousData()
{
  if (detectorBB)
  {
    delete detectorBB;
    detectorBB = NULL;
  }
}

void Detector::detect(const Mat &greyImg, const Mat &colImg)
{
    //For every bounding box, the output is confidence, pattern, variance

    cleanPreviousData();

    if(!initialised)
    {
        return;
    }

    //Prepare components
    dnnFilter->nextIterationSimple(colImg, frameNumber); //Calculates integral images
    cfClassifier->nextIteration(colImg, frameNumber); //Calculates integral images

    vector<Rect> faces = dnnFilter->faces;
    pair<Rect, float> res, max;

    #pragma omp parallel for

    for(int i = 0; i < faces.size(); i++)
    {
      res = cfClassifier->detect(faces[i]);
      if(res.second > max.second)
        max = res;
    }

    if(max.second > cfClassifier->detectionThreshold)
    {
      Rect tmp = max.first;
      detectorBB = new Rect(tmp.x, tmp.y, tmp.width, tmp.height);
      curConf = max.second;
    }

    frameNumber++;
}

} /* namespace tld */
