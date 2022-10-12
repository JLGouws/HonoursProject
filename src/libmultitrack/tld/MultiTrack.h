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
 * TLD.h
 *
 *  Created on: Nov 17, 2011
 *      Author: Georg Nebehay
 */

#ifndef MULTITRACK_H_
#define MULTITRACK_H_

#include <opencv/cv.h>
#include <vector>
#include <set>

#include "MedianFlowTracker.h"
#include "KCFTracker.h"
#include "Detector.h"

namespace tld
{

class MultiTrack
{
  typedef struct Target_t
  {
      cv::Rect *currBB,
               *prevBB;
      Detector *detector;
      KCF::KCFTracker *tracker;
      float currConf;
      bool valid,
           wasValid,
           learning;
      int targetNumber;
  } Target_t;

  public:
    bool trackerEnabled;
    bool detectorEnabled;
    bool learningEnabled;
    bool alternating;

//    MedianFlowTracker *medianFlowTracker;
//    KCF::KCFTracker *kcfTracker;
    std::vector<Target_t *> targets;
    cv::Mat prevImg;
    cv::Mat prevImgGrey;
    cv::Mat currImg;
    cv::Mat currImgGrey;

    MultiTrack();
    virtual ~MultiTrack();
    void release();
//    void selectObject(const cv::Mat &img, cv::Rect *bb);
    void init(const cv::Mat &img, cv::Rect *bb);
    void addTarget(const cv::Mat &im, cv::Rect *bb);
    void addTarget(cv::Rect *bb);
    void processImage(const cv::Mat &img);
    std::vector< std::pair<cv::Rect, std::pair<int, float>> > getResults();

    private:
      long frameNumber;

      void storeCurrentData();
      void storeCurrentTarget(Target_t *t);
      void fuseHypotheses(Target_t *t);
      void learn(Target_t *t);
  //    void initialLearning();
      void initialLearning(Target_t *t);
};

} /* namespace tld */
#endif /* MULTITRACK_H_ */
