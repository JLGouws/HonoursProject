#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <cstring>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <stdexcept>
#include <chrono>

#include "MultiTrack.h" 

/*
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
*/

using namespace std;
using namespace cv;
int main( int argc, char** argv ){
  FILE *out = fopen("output.txt", "w");
  // show help
  // declares all required variables
  dnn::Net detector = dnn::readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel");
  detector = dnn::readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel");
  Mat frame;
  Rect roi;
  int frameCount = 0;
  bool tracking = false;
  // create a tracker object
  tld::MultiTrack *tld = new tld::MultiTrack();
  // set input video
  VideoCapture cap("/home/jgouws/tldSourceCode/frames/TakiTaki1080p/%04d.png");

  cap >> frame;
  int imw;
  int imh;
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
  //193 -> 700
  while(frameCount < 192){
    cap >> frame;
    frameCount++;
    tld->processImage(frame);
  }

  std::vector<Rect> faces;

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
    //roi=selectROI("tracker",frame);

  float minConfidence = 0.15;

  imw = frame.cols;
  imh = frame.rows;
  cout << imw << "x" << imh << endl;
  Mat blob = dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false, false);

  detector.setInput(blob, "data");

  Mat results = detector.forward("detection_out");

  Mat detectionMat(results.size[2], results.size[3], CV_32F, results.ptr<float>());

  for(int i = 0; i < detectionMat.rows; i++)
  {
      float confidence = detectionMat.at<float>(i, 2);

      if(confidence > minConfidence)
      {
          int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * imw);
          int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * imh);
          int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * imw);
          int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * imh);

          Rect object((int)xLeftBottom, (int)yLeftBottom,
                      (int)(xRightTop - xLeftBottom),
                      (int)(yRightTop - yLeftBottom));

          tld->addTarget(&object);
          cout << object << endl;
      }
  }
  Rect object(557, 147,
              48, 61);

  tld->addTarget(&object);

  tracking = true;

  int i = 0;
  vector<pair<Rect, pair<int, float>>> targets;
  chrono::steady_clock::time_point begin, end;
  begin = chrono::steady_clock::now();
  for (;i <= 600; i++){
    /*
    if(frameCount == 163){
      imshow("tracker", frame);
      waitKey(10000);
    }*/
    // get frame from the video
    cap >> frame;
    frameCount++;
    // update the tracking result
    if(tracking) {
      tld->processImage(frame);
      targets.clear();
      targets = tld->getResults();

      for(int j = 0; j < targets.size(); j++)
      {
        Scalar color;
        switch(targets.at(j).second.first)
        {
          case 0:
            color = Scalar( 0, 255, 0);
          break;
          case 1:
            color = Scalar( 0, 0, 255);
          break;
          case 2:
            color = Scalar( 255, 0, 0);
          break;
          case 3:
            color = Scalar( 255, 0, 255);
          break;
        }
        rectangle(frame, targets.at(j).first, color, 2, 1 );
        Rect bb = targets.at(j).first;
        //fprintf(out, "%05d %d,%d,%d,%d,%d,%f\n", i, targets.at(j).second, bb.x, bb.y, bb.width, bb.height, targets.at(j).second.second);
      }
    }

    //fprintf(out, "\n");

    //imshow("tracker", frame);
    //imwrite(string_format("/home/jgouws/tldSourceCode/frames/tldOut/tldOUT%04d.jpg", i), frame);
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
    if(waitKey(10)==27)break;
  }
  
  end = chrono::steady_clock::now();
  cout << "Time difference: " << chrono::duration_cast<chrono::microseconds> (end - begin).count() << " microsecods" << endl;
  cout << "Frames: " << i << endl;
  cout << "fps: " << 1e6 * (double) i / chrono::duration_cast<chrono::microseconds> (end - begin).count()  << endl;
  cap.release();
  fclose(out);
  //writer.release();
  delete tld;

  return 0;
}
