#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <cstring>
#include <stdlib.h>
#include "videoHandler.hpp"

#include "MultiTrack.h"
#include "TLDUtil.h"

using namespace std;
using namespace cv;


int main(int argc, char** argv){

  tld::MultiTrack *tracker = new tld::MultiTrack();

  Mat image;

  string video;

  VideoCapture cap(0);
  // get bounding box

  vector<pair<Rect, pair<int, float>>> targets;

  Rect bb;
  for ( ;; ) {
    // get frame from the video
    cap >> image;
    // stop the program if no more images
    if(image.rows==0 || image.cols==0)
      break;

    tracker->processImage(image);

    targets.clear();
    targets = tracker->getResults();

    for(int i = 0; i < targets.size(); i++)
    {
      Scalar color = targets.at(i).second.first == 0 ? Scalar( 0, 255, 0) : Scalar( 255, 0, 0);
      rectangle(image, targets.at(i).first, color, 2, 1 );
    }

    imshow("tracker", image);

    //quit on ESC button
    if(waitKey(20)==27)break;
    if(waitKey(20)=='s')
    {
      Rect roi=selectROI("tracker", image);
      tracker->addTarget(&roi);
      std::cout << roi << std::endl;
    }
  }

  delete tracker;

  return 0;
}
