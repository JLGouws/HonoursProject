/*
 *  KCFTracker.cpp
 *
 *  Created on: Aug 17, 2022
 *      Author: J L Gouws
 */

/*
 * Yes this is really stupid but whatever
 */

#include "KCFTracker.h"
#include "kcf.h"

using namespace cv;

namespace KCF 
{
  class KCFTrackerImpl
  {
    public:
      KCFTrackerImpl(){};
      virtual ~KCFTrackerImpl(){};
      Rect *trackImpl(cv::Mat &img, cv::Rect *prevBB);
      void initImpl(const cv::Mat &img, const cv::Rect &bbox);
    private:
      KCF_Tracker tracker;
  };

  void KCFTrackerImpl::initImpl(const Mat &img, const Rect &bbox)
  {
    tracker.init(img, bbox);
  }

  Rect *KCFTrackerImpl::trackImpl(Mat &img, Rect *prevBB)
  {
      if(prevBB != NULL)
      {
          if(prevBB->width <= 0 || prevBB->height <= 0)                           
            return NULL; 

          tracker.track(img);

          BBox_c bb = tracker.getBBox();

          bool success = tracker.getMaxResponse() > 0.2;

          //Extract subimage
          float x, y, w, h;
          x = bb.cx - bb.w / 2.;
          y = bb.cy - bb.h / 2.;
          w = bb.w;
          h = bb.h;

          //TODO: Introduce a check for a minimum size
          if(!success || x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > img.cols || y + h > img.rows || x != x || y != y || w != w || h != h) //x!=x is check for nan
          {
            return NULL;
          }
          else
          {
              return new Rect(x, y, w, h);
          }
      }
      return NULL;
  }

  KCFTracker::KCFTracker()
  {
      trackerBB = NULL;
      pimpl = new KCFTrackerImpl();
  }

  KCFTracker::~KCFTracker()
  {
      cleanPreviousData();
      delete pimpl;
  }

  void KCFTracker::cleanPreviousData()
  {
      delete trackerBB;
      trackerBB = NULL;
  }

  void KCFTracker::init(const Mat &img, const Rect &bbox)
  {
    pimpl->initImpl(img, bbox);
  }

  void KCFTracker::track(Mat &img, Rect *prevBB)
  {
    trackerBB = pimpl->trackImpl(img, prevBB);
  }
}
