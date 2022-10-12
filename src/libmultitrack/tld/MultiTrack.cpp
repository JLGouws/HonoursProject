/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/
/*
 * TLD.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: Georg Nebehay
 */

#include "MultiTrack.h"

#include <iostream>

#include "CFClassifier.h"
#include "Patch.h"
#include "TLDUtil.h"
#include "KCFTracker.h"

using namespace std;
using namespace cv;
using namespace KCF;

namespace tld
{

MultiTrack::MultiTrack()
{
    trackerEnabled = true;
    detectorEnabled = true;
    learningEnabled = true;
    alternating = false;
    frameNumber = 0;
}

MultiTrack::~MultiTrack()
{
    storeCurrentData();

    for (auto const& t : targets) {
      if(t->detector)
      {
          delete t->detector;
          t->detector= NULL;
      }

      if(t->currBB)
      {
          delete t->currBB;
          t->currBB = NULL;
      }

      if(t->tracker)
      {
          delete t->tracker;
          t->tracker= NULL;
      }

      if(t->prevBB)
      {
          delete t->prevBB;
          t->prevBB = NULL;
      }

    }
    targets.clear();
}

void MultiTrack::release()
{
    for (auto const& t : targets) {
      t->detector->release();
      t->tracker->cleanPreviousData();//medianFlowTracker->cleanPreviousData();
      if(t->currBB)
      {
          delete t->currBB;
          t->currBB = NULL;
      }
    }

}

void MultiTrack::storeCurrentData()
{
    prevImgGrey.release();
    prevImgGrey = currImgGrey; //Store old image (if any)
    for (auto const& t : targets) {
      if(t->currBB)//Store old bounding box (if any)
      {
          t->prevBB->x = t->currBB->x;
          t->prevBB->y = t->currBB->y;
          t->prevBB->width = t->currBB->width;
          t->prevBB->height = t->currBB->height;
      }
      else
      {
          t->prevBB->x = 0;
          t->prevBB->y = 0;
          t->prevBB->width = 0;
          t->prevBB->height = 0;
      }

      t->detector->cleanPreviousData(); //Reset detector results
      t->tracker->cleanPreviousData();//medianFlowTracker->cleanPreviousData();

      t->wasValid = t->valid;
    }
}
void MultiTrack::storeCurrentTarget(Target_t *t)
{
    if(t->currBB)//Store old bounding box (if any)
    {
        t->prevBB->x = t->currBB->x;
        t->prevBB->y = t->currBB->y;
        t->prevBB->width = t->currBB->width;
        t->prevBB->height = t->currBB->height;
    }
    else
    {
        t->prevBB->x = 0;
        t->prevBB->y = 0;
        t->prevBB->width = 0;
        t->prevBB->height = 0;
    }

    t->detector->cleanPreviousData(); //Reset detector results
    t->tracker->cleanPreviousData();//medianFlowTracker->cleanPreviousData();

    t->wasValid = t->valid;
}

void MultiTrack::init(const Mat &img, Rect *bb)
{
    targets.clear();

    currImgGrey = Mat(img.rows, img.cols, CV_8UC1);         
    cvtColor(img, currImgGrey, CV_BGR2GRAY);

    addTarget(bb);
}

void MultiTrack::addTarget(const Mat &im, Rect *bb)
{
    Target_t *tg = new Target_t();
    tg->tracker = new KCFTracker();
    if(targets.size() == 0)
      tg->detector = new Detector(frameNumber);
    else
    {
      tg->detector = new Detector(targets.at(0)->detector->dnnFilter, frameNumber);
    }
    tg->detector->release();
    tg->detector->objWidth = bb->width;
    tg->detector->objHeight = bb->height;
    tg->detector->imgWidth = currImgGrey.cols;
    tg->detector->imgHeight = currImgGrey.rows;
    tg->detector->imgWidthStep = currImgGrey.step;
    tg->detector->init();

    tg->tracker->init(currImgGrey, *bb);
    tg->currBB = new Rect(bb->x, bb->y, bb->width, bb->height);
    tg->prevBB = new Rect(0, 0, 0, 0);
    tg->currConf = 1;
    tg->learning = false;
    tg->valid = true;
    tg->wasValid = false;
    tg->targetNumber = targets.size();
    targets.push_back(tg);

    Mat grey;

    currImgGrey = Mat(im.rows, im.cols, CV_8UC1);         
    cvtColor(im, grey, CV_BGR2GRAY);

    tg->detector->cfClassifier->learnPositive(im, *bb);
}

void MultiTrack::addTarget(Rect *bb)
{
    Target_t *tg = new Target_t();
    tg->tracker = new KCFTracker();
    if(targets.size() == 0)
      tg->detector = new Detector(frameNumber);
    else
    {
      tg->detector = new Detector(targets.at(0)->detector->dnnFilter, frameNumber);
    }
    tg->detector->release();
    tg->detector->objWidth = bb->width;
    tg->detector->objHeight = bb->height;
    tg->detector->imgWidth = currImgGrey.cols;
    tg->detector->imgHeight = currImgGrey.rows;
    tg->detector->imgWidthStep = currImgGrey.step;
    tg->detector->init();
    tg->tracker->init(currImgGrey, *bb);
    tg->currBB = new Rect(bb->x, bb->y, bb->width, bb->height);
    tg->prevBB = new Rect(0, 0, 0, 0);
    tg->currConf = 1;
    tg->learning = false;
    tg->valid = true;
    tg->wasValid = false;
    tg->targetNumber = targets.size();
    targets.push_back(tg);

    initialLearning(tg);
}

void MultiTrack::processImage(const Mat &img)
{
    prevImgGrey.release();
    prevImgGrey = currImgGrey; //Store old image (if any)
    prevImg = currImg;
    Mat grey_frame;
    currImg = img;
    cvtColor(img, grey_frame, CV_BGR2GRAY);
    currImgGrey = grey_frame; // Store new image , right after storeCurrentData();

    for (auto const& t : targets) {
      storeCurrentTarget(t);

      if(trackerEnabled)
      {
          t->tracker->track(currImgGrey, t->prevBB);//t->tracker->track(currImg, t->prevBB);//medianFlowTracker->track(prevImgGrey, currImg, prevBB);
      }

      if(detectorEnabled)// && (!alternating || t->tracker->trackerBB == NULL))//medianFlowTracker->trackerBB == NULL))
      {
          t->detector->detect(grey_frame, currImg);//, t->tracker->trackerBB);
      }

      fuseHypotheses(t);

      learn(t);
    }

    frameNumber++;
}

vector<pair<Rect, pair<int, float>>> MultiTrack::getResults()
{
  vector<pair<Rect, pair<int, float>>> results;
  for (auto const& t : targets) {
    if(t->currBB != NULL)
      results.push_back(pair<Rect, pair<int, float>>(*t->currBB, pair<int, float>(t->targetNumber, t->currConf)));
  }
  return results;
}

void MultiTrack::fuseHypotheses(Target_t *t)
{
    Rect *trackerBB = t->tracker->trackerBB;//Rect *trackerBB = medianFlowTracker->trackerBB;
    Rect *detectorBB = t->detector->detectorBB;

    if(t->currBB)
    {
        delete t->currBB;
        t->currBB = NULL;
    }
    t->currConf = 0;
    t->valid = false;

    float confDetector = 0;

    if(detectorBB)
    {
        confDetector = t->detector->curConf;
    }

    if(trackerBB != NULL)
    {
        float confTracker = t->detector->cfClassifier->classifyWindow(currImg, *trackerBB);
        if(t->currBB)
        {
            delete t->currBB;
            t->currBB = NULL;
        }

        if(detectorBB && confDetector > confTracker && tldOverlapRectRect(*trackerBB, *detectorBB) < 0.5)
        {
            t->currBB = tldCopyRect(detectorBB);
            t->currConf = confDetector;
        }
        else
        {
            t->currBB = tldCopyRect(trackerBB);
            t->currConf = confTracker;

            t->valid = true;
        }
    }
    else if(detectorBB)
    {
        if(t->currBB)
        {
            delete t->currBB;
            t->currBB = NULL;
        }

        float maxOverlap = 0,
              overlap = 0;

        for (auto const& tg : targets) 
        {
          if (tg->currBB) {
            overlap = tldOverlapRectRect(*tg->currBB, *detectorBB);
            maxOverlap = overlap > maxOverlap ? overlap : maxOverlap;
          }
        }

        if (maxOverlap < 0.1)
        {
          t->currBB = tldCopyRect(detectorBB);
          t->currConf = confDetector;
          t->valid = true;
          //t->tracker->place(*detectorBB);
          delete t->tracker;
          t->tracker = new KCFTracker();
          t->tracker->init(currImgGrey, *detectorBB);
        }
    }

    /*
    float var = CalculateVariance(patch.values, nn->patch_size*nn->patch_size);

    if(var < min_var) { //TODO: Think about incorporating this
        printf("%f, %f: Variance too low \n", var, classifier->min_var);
        valid = 0;
    }*/
}

void MultiTrack::initialLearning(Target_t *t)
{
    t->learning = true; //This is just for display purposes

    t->detector->detect(currImgGrey, currImg);

    //This is the positive patch

    float IoU;

    //Add all bounding boxes with high overlap

    vector<Rect> faces = t->detector->dnnFilter->faces;
    vector<Patch> patches;

    //First: Find overlapping positive and negative patches

    for(int i = 0; i < faces.size(); i++)
    {
        IoU = tldOverlapRectRect(faces[i], *(t->currBB));
        if(IoU > 0.6)
        {
          Patch p;
          p.positive = 1;
          p.roi = faces[i];
          patches.push_back(p);
        }
        else
        {
          Patch p;
          p.positive = 0;
          p.roi = faces[i];
          patches.push_back(p);
        }
    }

    srand(1); //TODO: This is not guaranteed to affect random_shuffle

    random_shuffle(patches.begin(), patches.end());

    t->detector->cfClassifier->learn(currImg, patches);
    t->detector->cfClassifier->learnPositive(currImg, *t->currBB);

}
/*
void MultiTrack::initialLearning()
{
    learning = true; //This is just for display purposes

    DetectionResult *detectionResult = detectorCascade->detectionResult;

    detectorCascade->detect(currImgGrey);

    //This is the positive patch
    NormalizedPatch patch;
    tldExtractNormalizedPatchRect(currImgGrey, currBB, patch.values);
    patch.positive = 1;

    float initVar = tldCalcVariance(patch.values, TLD_PATCH_SIZE * TLD_PATCH_SIZE);
    detectorCascade->varianceFilter->minVar = initVar / 2;


    float *overlap = new float[detectorCascade->numWindows];
    tldOverlapRect(detectorCascade->windows, detectorCascade->numWindows, currBB, overlap);

    //Add all bounding boxes with high overlap

    vector< pair<int, float> > positiveIndices;
    vector<int> negativeIndices;

    //First: Find overlapping positive and negative patches

    for(int i = 0; i < detectorCascade->numWindows; i++)
    {

        if(overlap[i] > 0.6)
        {
            positiveIndices.push_back(pair<int, float>(i, overlap[i]));
        }

        if(overlap[i] < 0.2)
        {
            float variance = detectionResult->variances[i];

            if(!detectorCascade->varianceFilter->enabled || variance > detectorCascade->varianceFilter->minVar)   //TODO: This check is unnecessary if minVar would be set before calling detect.
            {
                negativeIndices.push_back(i);
            }
        }
    }

    sort(positiveIndices.begin(), positiveIndices.end(), tldSortByOverlapDesc);

    vector<NormalizedPatch> patches;

    patches.push_back(patch); //Add first patch to patch list

    int numIterations = std::min<size_t>(positiveIndices.size(), 10); //Take at most 10 bounding boxes (sorted by overlap)

    for(int i = 0; i < numIterations; i++)
    {
        int idx = positiveIndices.at(i).first;
        //Learn this bounding box
        //TODO: Somewhere here image warping might be possible
        detectorCascade->ensembleClassifier->learn(&detectorCascade->windows[TLD_WINDOW_SIZE * idx], true, &detectionResult->featureVectors[detectorCascade->numTrees * idx]);
    }

    srand(1); //TODO: This is not guaranteed to affect random_shuffle

    random_shuffle(negativeIndices.begin(), negativeIndices.end());

    //Choose 100 random patches for negative examples
    for(size_t i = 0; i < std::min<size_t>(100, negativeIndices.size()); i++)
    {
        int idx = negativeIndices.at(i);

        NormalizedPatch patch;
        tldExtractNormalizedPatchBB(currImgGrey, &detectorCascade->windows[TLD_WINDOW_SIZE * idx], patch.values);
        patch.positive = 0;
        patches.push_back(patch);
    }

    detectorCascade->nnClassifier->learn(patches);

    delete[] overlap;

}
*/

//Do this when current trajectory is valid
void MultiTrack::learn(Target_t *t)
{
    //This is the positive patch

    float IoU;

    //Add all bounding boxes with high overlap

    vector<Rect> faces = t->detector->dnnFilter->faces;
    vector<Patch> patches;

    //First: Find overlapping positive and negative patches

    for(int i = 0; i < faces.size(); i++)
    {
        IoU = tldOverlapRectRect(faces[i], *(t->currBB));
        if(IoU > 0.6)
        {
          Patch p;
          p.positive = 1;
          p.roi = faces[i];
          patches.push_back(p);
        }
        else
        {
          Patch p;
          p.positive = 0;
          p.roi = faces[i];
          patches.push_back(p);
        }
    }

    srand(1); //TODO: This is not guaranteed to affect random_shuffle

    random_shuffle(patches.begin(), patches.end());

    t->detector->cfClassifier->learn(currImg, patches);
    t->detector->cfClassifier->learnPositive(currImg, *t->currBB);
}

typedef struct
{
    int index;
    int P;
    int N;
} TldExportEntry;


static int fpeek(FILE *stream)
{
  int c = fgetc(stream);
  ungetc(c, stream);
  return c;
}


} /* namespace tld */
