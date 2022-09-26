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

#include "NNClassifier.h"
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
      if(t->detectorCascade)
      {
          delete t->detectorCascade;
          t->detectorCascade = NULL;
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
      t->detectorCascade->release();
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

      t->detectorCascade->cleanPreviousData(); //Reset detector results
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

    t->detectorCascade->cleanPreviousData(); //Reset detector results
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
      tg->detectorCascade = new DetectorCascade(frameNumber);
    else
    {
      tg->detectorCascade = new DetectorCascade(targets.at(0)->detectorCascade->dnnFilter, frameNumber);
    }
    tg->detectorCascade->release();
    tg->detectorCascade->objWidth = bb->width;
    tg->detectorCascade->objHeight = bb->height;
    tg->detectorCascade->imgWidth = currImgGrey.cols;
    tg->detectorCascade->imgHeight = currImgGrey.rows;
    tg->detectorCascade->imgWidthStep = currImgGrey.step;
    if(targets.size() == 0)
      tg->detectorCascade->init();
    else
    {
      DetectorCascade *dc = targets.at(0)->detectorCascade;
      tg->detectorCascade->init(dc->numWindows, dc->windows, dc->windowOffsets, dc->numScales, dc->scales);
    }
    tg->nnClassifier = tg->detectorCascade->nnClassifier;
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

    NormalizedPatch patch;
    patch.positive = 1;
    tldExtractNormalizedPatchRect(grey, bb, patch.values);

    vector<NormalizedPatch> patches;

    patches.push_back(patch);

    tg->detectorCascade->nnClassifier->learn(patches);
}

void MultiTrack::addTarget(Rect *bb)
{
    Target_t *tg = new Target_t();
    tg->tracker = new KCFTracker();
    if(targets.size() == 0)
      tg->detectorCascade = new DetectorCascade(frameNumber);
    else
    {
      tg->detectorCascade = new DetectorCascade(targets.at(0)->detectorCascade->dnnFilter, frameNumber);
    }
    tg->detectorCascade->release();
    tg->detectorCascade->objWidth = bb->width;
    tg->detectorCascade->objHeight = bb->height;
    tg->detectorCascade->imgWidth = currImgGrey.cols;
    tg->detectorCascade->imgHeight = currImgGrey.rows;
    tg->detectorCascade->imgWidthStep = currImgGrey.step;
    if(targets.size() == 0)
      tg->detectorCascade->init();
    else
    {
      DetectorCascade *dc = targets.at(0)->detectorCascade;
      tg->detectorCascade->init(dc->numWindows, dc->windows, dc->windowOffsets, dc->numScales, dc->scales);
    }
    tg->nnClassifier = tg->detectorCascade->nnClassifier;
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
          t->detectorCascade->detect(grey_frame, currImg);//, t->tracker->trackerBB);
      }

      fuseHypotheses(t);

      learn(t);
    }

    frameNumber++;
}

vector<pair<Rect, int>> MultiTrack::getResults()
{
  vector<pair<Rect, int>> results;
  for (auto const& t : targets) {
    if(t->currBB != NULL)
      results.push_back(pair<Rect, int>(*t->currBB, t->targetNumber));
  }
  return results;
}

void MultiTrack::fuseHypotheses(Target_t *t)
{
    Rect *trackerBB = t->tracker->trackerBB;//Rect *trackerBB = medianFlowTracker->trackerBB;
    int numClusters = t->detectorCascade->detectionResult->numClusters;
    Rect *detectorBB = t->detectorCascade->detectionResult->detectorBB;

    if(t->currBB)
    {
        delete t->currBB;
        t->currBB = NULL;
    }
    t->currConf = 0;
    t->valid = false;

    float confDetector = 0;

    if(numClusters == 1)
    {
        confDetector = t->nnClassifier->classifyBB(currImgGrey, detectorBB);
    }

    if(trackerBB != NULL)
    {
        float confTracker = t->nnClassifier->classifyBB(currImgGrey, trackerBB);
        if(t->currBB)
        {
            delete t->currBB;
            t->currBB = NULL;
        }

        if(numClusters == 1 && confDetector > confTracker && tldOverlapRectRect(*trackerBB, *detectorBB) < 0.5)
        {
            t->currBB = tldCopyRect(detectorBB);
            t->currConf = confDetector;
        }
        else
        {
            t->currBB = tldCopyRect(trackerBB);
            t->currConf = confTracker;

            if(confTracker > t->nnClassifier->thetaTP)
            {
                t->valid = true;
            }
            else if(t->wasValid && confTracker > t->nnClassifier->thetaFP)
            {
                t->valid = true;
            }
        }
    }
    else if(numClusters == 1)
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

    DetectionResult *detectionResult = t->detectorCascade->detectionResult;

    t->detectorCascade->detect(currImgGrey, currImg);

    //This is the positive patch
    NormalizedPatch patch;
    tldExtractNormalizedPatchRect(currImgGrey, t->currBB, patch.values);
    patch.positive = 1;

    float *overlap = new float[t->detectorCascade->numWindows];
    tldOverlapRect(t->detectorCascade->windows, t->detectorCascade->numWindows, t->currBB, overlap);

    //Add all bounding boxes with high overlap

    vector< pair<int, float>> positiveIndices;
    vector<int> negativeIndices;

    //First: Find overlapping positive and negative patches

    for(int i = 0; i < t->detectorCascade->numWindows; i++)
    {

        if(overlap[i] > 0.6)
        {
            positiveIndices.push_back(pair<int, float>(i, overlap[i]));
        }
        else if (overlap[i] > 0.6)
        {
            //cout << " negative example" << endl;
            negativeIndices.push_back(i);
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
        t->detectorCascade->ensembleClassifier->learn(&t->detectorCascade->windows[TLD_WINDOW_SIZE * idx], true, &detectionResult->featureVectors[t->detectorCascade->numTrees * idx]);
    }

    srand(1); //TODO: This is not guaranteed to affect random_shuffle

    random_shuffle(negativeIndices.begin(), negativeIndices.end());

    //Choose 100 random patches for negative examples
    for(size_t i = 0; i < std::min<size_t>(100, negativeIndices.size()); i++)
    {
        int idx = negativeIndices.at(i);

        NormalizedPatch patch;
        tldExtractNormalizedPatchBB(currImgGrey, &t->detectorCascade->windows[TLD_WINDOW_SIZE * idx], patch.values);
        patch.positive = 0;
        patches.push_back(patch);
    }

    t->detectorCascade->nnClassifier->learn(patches);

    delete[] overlap;

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
    if(!learningEnabled || !t->valid || !detectorEnabled)
    {
        t->learning = false;
        return;
    }

    t->learning = true;

    DetectionResult *detectionResult = t->detectorCascade->detectionResult;

    if(!detectionResult->containsValidData)
    {
        t->detectorCascade->detect(currImgGrey, currImg);
    }

    //This is the positive patch
    NormalizedPatch patch;
    tldExtractNormalizedPatchRect(currImgGrey, t->currBB, patch.values);

    float *overlap = new float[t->detectorCascade->numWindows];
    tldOverlapRect(t->detectorCascade->windows, t->detectorCascade->numWindows, t->currBB, overlap);

    //Add all bounding boxes with high overlap

    vector<pair<int, float> > positiveIndices;
    vector<int> negativeIndices;
    vector<int> negativeIndicesForNN;

    //First: Find overlapping positive and negative patches

    for(int i = 0; i < t->detectorCascade->numWindows; i++)
    {

        if(overlap[i] > 0.6)
        {
            positiveIndices.push_back(pair<int, float>(i, overlap[i]));
        }

        if(overlap[i] < 0.2)
        {
            if(!t->detectorCascade->ensembleClassifier->enabled || detectionResult->posteriors[i] > 0.5)   //Should be 0.5 according to the paper
            {
                negativeIndices.push_back(i);
            }

            if(!t->detectorCascade->ensembleClassifier->enabled || detectionResult->posteriors[i] > 0.5)
            {
                negativeIndicesForNN.push_back(i);
            }

        }
    }

    sort(positiveIndices.begin(), positiveIndices.end(), tldSortByOverlapDesc);

    vector<NormalizedPatch> patches;

    patch.positive = 1;
    patches.push_back(patch);
    //TODO: Flip

    int numIterations = std::min<size_t>(positiveIndices.size(), 10); //Take at most 10 bounding boxes (sorted by overlap)

    for(size_t i = 0; i < negativeIndices.size(); i++)
    {
        int idx = negativeIndices.at(i);
        //TODO: Somewhere here image warping might be possible
        t->detectorCascade->ensembleClassifier->learn(&t->detectorCascade->windows[TLD_WINDOW_SIZE * idx], false, &detectionResult->featureVectors[t->detectorCascade->numTrees * idx]);
    }

    //TODO: Randomization might be a good idea
    for(int i = 0; i < numIterations; i++)
    {
        int idx = positiveIndices.at(i).first;
        //TODO: Somewhere here image warping might be possible
        t->detectorCascade->ensembleClassifier->learn(&t->detectorCascade->windows[TLD_WINDOW_SIZE * idx], true, &detectionResult->featureVectors[t->detectorCascade->numTrees * idx]);
    }

    for(size_t i = 0; i < negativeIndicesForNN.size(); i++)
    {
        int idx = negativeIndicesForNN.at(i);

        NormalizedPatch patch;
        tldExtractNormalizedPatchBB(currImgGrey, &t->detectorCascade->windows[TLD_WINDOW_SIZE * idx], patch.values);
        patch.positive = 0;
        patches.push_back(patch);
    }

    t->detectorCascade->nnClassifier->learn(patches);

    //cout << "NN " << t->targetNumber << " has now " << t->detectorCascade->nnClassifier->truePositives->size() << " positives and " << t->detectorCascade->nnClassifier->falsePositives->size() << " negatives.\n";

    delete[] overlap;
    //cout << endl << endl;
}

typedef struct
{
    int index;
    int P;
    int N;
} TldExportEntry;

void MultiTrack::writeToFile(const char *path)
{
    NNClassifier *nn;       
    EnsembleClassifier *ec; 

    FILE *file = fopen(path, "w");
    fprintf(file, "#Tld ModelExport\n");
    for (auto const& t : targets) 
    {
        nn = t->detectorCascade->nnClassifier;
        ec = t->detectorCascade->ensembleClassifier;
        fprintf(file, "%d #Target number\n", t->targetNumber);
        fprintf(file, "%d #width\n", t->detectorCascade->objWidth);
        fprintf(file, "%d #height\n", t->detectorCascade->objHeight);
        fprintf(file, "%f #dnn_min_conf\n", t->detectorCascade->dnnFilter->minConfidence);
        fprintf(file, "%d #Positive Sample Size\n", nn->truePositives->size());



        for(size_t s = 0; s < nn->truePositives->size(); s++)
        {
            float *imageData = nn->truePositives->at(s).values;

            for(int i = 0; i < TLD_PATCH_SIZE; i++)
            {
                for(int j = 0; j < TLD_PATCH_SIZE; j++)
                {
                    fprintf(file, "%f ", imageData[i * TLD_PATCH_SIZE + j]);
                }

                fprintf(file, "\n");
            }
        }

        fprintf(file, "%d #Negative Sample Size\n", nn->falsePositives->size());

        for(size_t s = 0; s < nn->falsePositives->size(); s++)
        {
            float *imageData = nn->falsePositives->at(s).values;

            for(int i = 0; i < TLD_PATCH_SIZE; i++)
            {
                for(int j = 0; j < TLD_PATCH_SIZE; j++)
                {
                    fprintf(file, "%f ", imageData[i * TLD_PATCH_SIZE + j]);
                }

                fprintf(file, "\n");
            }
        }

        fprintf(file, "%d #numtrees\n", ec->numTrees);
        t->detectorCascade->numTrees = ec->numTrees;
        fprintf(file, "%d #numFeatures\n", ec->numFeatures);
        t->detectorCascade->numFeatures = ec->numFeatures;

        for(int i = 0; i < ec->numTrees; i++)
        {
            fprintf(file, "#Tree %d\n", i);

            for(int j = 0; j < ec->numFeatures; j++)
            {
                float *features = ec->features + 4 * ec->numFeatures * i + 4 * j;
                fprintf(file, "%f %f %f %f # Feature %d\n", features[0], features[1], features[2], features[3], j);
            }

            //Collect indices
            vector<TldExportEntry> list;

            for(int index = 0; index < pow(2.0f, ec->numFeatures); index++)
            {
                int p = ec->positives[i * ec->numIndices + index];

                if(p != 0)
                {
                    TldExportEntry entry;
                    entry.index = index;
                    entry.P = p;
                    entry.N = ec->negatives[i * ec->numIndices + index];
                    list.push_back(entry);
                }
            }

            fprintf(file, "%d #numLeaves\n", list.size());

            for(size_t j = 0; j < list.size(); j++)
            {
                TldExportEntry entry = list.at(j);
                fprintf(file, "%d %d %d\n", entry.index, entry.P, entry.N);
            }
        }
    }
    fclose(file);

}

static int fpeek(FILE *stream)
{
  int c = fgetc(stream);
  ungetc(c, stream);
  return c;
}

void MultiTrack::readFromFile(const char *path)
{
    release();

    FILE *file = fopen(path, "r");

    if(file == NULL)
    {
        printf("Error: Model not found: %s\n", path);
        exit(1);
    }

    int MAX_LEN = 255;
    char str_buf[255];
    fgets(str_buf, MAX_LEN, file); /*Skip line*/

    while (fpeek(file) != EOF)
    {
      Target_t *t = new Target_t();

      t->detectorCascade = new DetectorCascade();

      NNClassifier *nn = t->detectorCascade->nnClassifier;
      EnsembleClassifier *ec = t->detectorCascade->ensembleClassifier;

      fscanf(file, "%d \n", &t->targetNumber);
      fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/
      fscanf(file, "%d \n", &t->detectorCascade->objWidth);
      fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/
      fscanf(file, "%d \n", &t->detectorCascade->objHeight);
      fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

      fscanf(file, "%f \n", &t->detectorCascade->dnnFilter->minConfidence);
      fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

      int numPositivePatches;
      fscanf(file, "%d \n", &numPositivePatches);
      fgets(str_buf, MAX_LEN, file); /*Skip line*/


      for(int s = 0; s < numPositivePatches; s++)
      {
          NormalizedPatch patch;

          for(int i = 0; i < 15; i++)   //Do 15 times
          {

              fgets(str_buf, MAX_LEN, file); /*Read sample*/

              char *pch;
              pch = strtok(str_buf, " \n");
              int j = 0;

              while(pch != NULL)
              {
                  float val = atof(pch);
                  patch.values[i * TLD_PATCH_SIZE + j] = val;

                  pch = strtok(NULL, " \n");

                  j++;
              }
          }

          nn->truePositives->push_back(patch);
      }

      int numNegativePatches;
      fscanf(file, "%d \n", &numNegativePatches);
      fgets(str_buf, MAX_LEN, file); /*Skip line*/


      for(int s = 0; s < numNegativePatches; s++)
      {
          NormalizedPatch patch;

          for(int i = 0; i < 15; i++)   //Do 15 times
          {

              fgets(str_buf, MAX_LEN, file); /*Read sample*/

              char *pch;
              pch = strtok(str_buf, " \n");
              int j = 0;

              while(pch != NULL)
              {
                  float val = atof(pch);
                  patch.values[i * TLD_PATCH_SIZE + j] = val;

                  pch = strtok(NULL, " \n");

                  j++;
              }
          }

          nn->falsePositives->push_back(patch);
      }

      fscanf(file, "%d \n", &ec->numTrees);
      t->detectorCascade->numTrees = ec->numTrees;
      fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

      fscanf(file, "%d \n", &ec->numFeatures);
      t->detectorCascade->numFeatures = ec->numFeatures;
      fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

      int size = 2 * 2 * ec->numFeatures * ec->numTrees;
      ec->features = new float[size];
      ec->numIndices = pow(2.0f, ec->numFeatures);
      ec->initPosteriors();

      for(int i = 0; i < ec->numTrees; i++)
      {
          fgets(str_buf, MAX_LEN, file); /*Skip line*/

          for(int j = 0; j < ec->numFeatures; j++)
          {
              float *features = ec->features + 4 * ec->numFeatures * i + 4 * j;
              fscanf(file, "%f %f %f %f", &features[0], &features[1], &features[2], &features[3]);
              fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/
          }

          /* read number of leaves*/
          int numLeaves;
          fscanf(file, "%d \n", &numLeaves);
          fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

          for(int j = 0; j < numLeaves; j++)
          {
              TldExportEntry entry;
              fscanf(file, "%d %d %d \n", &entry.index, &entry.P, &entry.N);
              ec->updatePosterior(i, entry.index, 1, entry.P);
              ec->updatePosterior(i, entry.index, 0, entry.N);
          }
      }

      t->detectorCascade->initWindowsAndScales();
      t->detectorCascade->initWindowOffsets();

      t->detectorCascade->propagateMembers();

      t->detectorCascade->initialised = true;

      ec->initFeatureOffsets();
    }

    fclose(file);
}


} /* namespace tld */
