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
 * MedianFlowTracker.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: Georg Nebehay
 */

#include "KCFTracker.h"
#include "kcf.cpp"

using namespace cv;

namespace KCF 
{
  KCF_Tracker tracker;

  KCFTracker::KCFTracker()
  {
      trackerBB = NULL;
  }

  KCFTracker::~KCFTracker()
  {
      cleanPreviousData();
  }

  void KCFTracker::cleanPreviousData()
  {
      delete trackerBB;
      trackerBB = NULL;
  }

  void KCFTracker::init(const Mat &img, const Rect &bbox)
  {
    tracker.init(img, bbox);
  }

  void KCFTracker::track(Mat &img, Rect *prevBB)
  {
      if(prevBB != NULL)
      {
          if(prevBB->width <= 0 || prevBB->height <= 0)                           
            return;                                                             

          tracker.track(img);

          BBox_c bb = tracker.getBBox();

          bool success = tracker.getMaxResponse() > 0.01;

          //Extract subimage
          float x, y, w, h;
          x = bb.cx - bb.w / 2.;
          y = bb.cy - bb.h / 2.;
          w = bb.w;
          h = bb.h;

          //TODO: Introduce a check for a minimum size
          if(!success || x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > img.cols || y + h > img.rows || x != x || y != y || w != w || h != h) //x!=x is check for nan
          {
              //Leave it empty
          }
          else
          {
              trackerBB = new Rect(x, y, w, h);
          }
      }
  }
}
