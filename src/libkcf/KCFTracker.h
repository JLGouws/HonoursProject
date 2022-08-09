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
 * MedianFlowTracker.h
 *
 *  Created on: Aug 1, 2022
 *      Author: J L Gouws
 */

#ifndef KCFTRACKER_H_
#define KCFTRACKER_H_

#include <opencv2/opencv.hpp>

namespace KCF {
class KCFTracker
{
  public:
    cv::Rect *trackerBB;

    KCFTracker();
    virtual ~KCFTracker();
    void cleanPreviousData();
    void track(cv::Mat &img, cv::Rect *prevBB);
    void init(const cv::Mat &img, const cv::Rect &bbox);
};
}

#endif /* KCFTRACKER_H_ */
