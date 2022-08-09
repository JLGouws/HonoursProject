#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <string>
#include <stdexcept>
#include <chrono>

#include "TLD.h" 

//https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

using namespace std;
using namespace cv;
int main( int argc, char** argv ){
  // show help
  // declares all required variables
  Mat frame;
  Rect roi;
  int frameCount = 0;
  bool tracking = false;
  // create a tracker object
  tld::TLD *tld = new tld::TLD();
  // set input video
  VideoCapture cap("/home/jgouws/tldSourceCode/frames/TakiTaki/%04d.jpg");

  cap >> frame;
  /*
  roi = selectROI("tracker",frame);
  //quit if ROI was not selected
  if(roi.width==0 || roi.height==0)
    return 0;
  // initialize the tracker
  tracker->init(frame,roi);
  */
  // perform the tracking process
  printf("Start the tracking process, press ESC to quit.\n");
  while(frameCount < 162){
    cap >> frame;
    frameCount++;
  }

  Mat grey(frame.rows, frame.cols, CV_8UC1);
  cvtColor(frame, grey, CV_BGR2GRAY);

  tld->detectorCascade->imgWidth = grey.cols;
  tld->detectorCascade->imgHeight = grey.rows;
  tld->detectorCascade->imgWidthStep = grey.step;

  /*
  bool isColor = (frame.type() == CV_8UC3);
  VideoWriter writer;
  int codec = VideoWriter::fourcc('M', 'P', '4', 'V');
  double fps = 20.0;
  string filename = "tldKcf.mp4";
  Size sizeFrame(640, 480);
  writer.open(filename, codec, fps, sizeFrame, isColor);
  if (!writer.isOpened())
    cout << "writer not opened" << endl;
  else
    cout << "writer opened" << endl;
  */
  int i = 0;
  chrono::steady_clock::time_point begin, end;
  for (;i <= 350; i++){
    /*
    if(frameCount == 163){
      imshow("tracker", frame);
      waitKey(10000);
    }*/
    if(frameCount == 162) {
      //roi=selectROI("tracker",frame);
      roi = Rect(256, 142, 20, 25);
      cout << roi;
      begin = chrono::steady_clock::now();
      tld->selectObject(grey, &roi);
      tracking = true;
      i = 1;
    }
    // get frame from the video
    cap >> frame;
    frameCount++;
    // update the tracking result
    if(tracking) {
      tld->processImage(frame);
      if(tld-> currBB != NULL)
      {
        roi = Rect(tld->currBB->x, tld->currBB->y, tld->currBB->width, tld->currBB->height);
        rectangle(frame, roi, Scalar( 0, 255, 0), 2, 1 );
      }
    }
    imshow("tracker", frame);
    imwrite(string_format("/home/jgouws/tldSourceCode/frames/tldOut/tldOUT%04d.jpg", i), frame);
    /*
    // show image with the tracked object
    if(waitKey(5)==115) {
      roi=selectROI("tracker",frame);
      cout << frameCount << endl;
      cout << roi << endl;
      tracker = TrackerTLD::create();
      tracker->init(frame,roi);
    }
    */
    //quit on ESC button
    //if(waitKey(20)==27)break;
  }
  
  end = chrono::steady_clock::now();
  cout << "Time difference: " << chrono::duration_cast<chrono::microseconds> (end - begin).count() << " microsecods" << endl;
  cout << "Frames: " << i << endl;
  cout << "fps: " << 1e6 * (double) i / chrono::duration_cast<chrono::microseconds> (end - begin).count()  << endl;
  cap.release();
  //writer.release();
  delete tld;

  return 0;
}
