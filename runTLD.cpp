#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <cstring>
#include <stdlib.h>
#include "videoHandler.hpp"

#include "tldTracker.hpp"
#include "kcfTracker.hpp"

using namespace std;
using namespace cv;


void playVideo(VidInfo *videoInfo) {
  Mat frame;
  for ( ;; ){
    // get frame from the video
    0[videoInfo->video] >> frame;
    // stop the program if no more images
    if(frame.rows==0 || frame.cols==0)
      break;
    rectangle( frame, *(videoInfo->initRoi), Scalar( 255, 0, 0 ), 2, 1 );
    imshow("tracker",frame);
    //quit on ESC button
    if(waitKey(2)==27)break;
  }
}

int main(int argc, char** argv){
  string basePath = "/home/jgouws/OTBData/";
  VidInfo *videoInfo;

  Ptr<Tracker> tracker = TrackerTLD::create();

  Mat image;

  string video;


  if(argc == 2) video = argv[1];
  videoInfo = load_video_info(basePath, video);
  0[videoInfo->video] >> image;
  tracker->init(image, 0[videoInfo->initRoi]);
//  cout << 0[videoInfo->initRoi].width << endl;
//  cout << 0[videoInfo->initRoi].height << endl;

  Rect2d bb;
  for ( ;; ) {
    // get frame from the video
    0[videoInfo->video] >> image;
    // stop the program if no more images
    if(image.rows==0 || image.cols==0)
      break;

    if(tracker->update(image, bb))
      rectangle(image, bb, Scalar( 255, 0, 0 ), 2, 1 );

    imshow("tracker", image);
    //quit on ESC button
    if(waitKey(2)==27)break;
  }


  //free memory of video info
  delete videoInfo->initRoi;
  delete videoInfo->video;
  free(videoInfo);

  return 0;
}
