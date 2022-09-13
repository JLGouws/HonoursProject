/*
 * KCFTracker.h
 *
 *  Created on: Aug 1, 2022
 *      Author: J L Gouws
 */

#ifndef KCFTRACKER_H_
#define KCFTRACKER_H_

#include <opencv2/opencv.hpp>

namespace KCF {

class KCFTrackerImpl;

class KCFTracker
{
  public:
    cv::Rect *trackerBB;

    KCFTracker();
    virtual ~KCFTracker();
    void cleanPreviousData();
    void track(cv::Mat &img, cv::Rect *prevBB);
    void init(const cv::Mat &img, const cv::Rect &bbox);
    void place(const cv::Rect &bbox);
  private:
    KCFTrackerImpl *pimpl;
};
}

#endif /* KCFTRACKER_H_ */
