#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>

#include <tldTracker.hpp>

using namespace std;
using namespace cv;
int main( int argc, char** argv ){
  // show help
  // declares all required variables
  Mat frame;
  Rect2d roi;
  // create a tracker object
  Ptr<Tracker> tracker = TrackerTLD::create();
  // set input video
  VideoCapture cap("../frames/TakiTaki/%04d.jpg");

  cap >> frame;

  roi = selectROI("tracker",frame);
  //quit if ROI was not selected
  if(roi.width==0 || roi.height==0)
    return 0;
  // initialize the tracker
  tracker->init(frame,roi);
  // perform the tracking process
  printf("Start the tracking process, press ESC to quit.\n");
  for ( ;; ){
    // get frame from the video
    cap >> frame;
    // update the tracking result
    if(tracker->update(frame,roi))
      rectangle(frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
    // show image with the tracked object
    imshow("tracker",frame);
    if(waitKey(5)==115) {
      roi=selectROI("tracker",frame);
      tracker = TrackerTLD::create();
      tracker->init(frame,roi);
    }
    //quit on ESC button
    if(waitKey(20)==27)break;
  }

  return 0;
}
