#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <cstring>
#include <stdlib.h>
#include "videoHandler.hpp"

#include "TLD.h"
#include "TLDUtil.h"

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

  tld::TLD *tracker = new tld::TLD();

  Mat image;

  string video;

  double totalOverlap = 0.;
  int numFrame = 0;
  float tP = 0, fP = 0, tN = 0, fN = 0;
  float currOverlap;


  if(argc == 2) video = argv[1];
  videoInfo = load_video_info(basePath, video);
  if(argv[1][0] == 'D' && argv[1][1] == 'a' && argv[1][2] == 'v') {
    cout << "David" << endl;
    for(int i = 0; i < 299; i++)
      0[videoInfo->video] >> image;
  }
  0[videoInfo->video] >> image;

  Mat grey(image.rows, image.cols, CV_8UC1);
  if(image.channels() == 3)
    cvtColor(image, grey, CV_BGR2GRAY);
  else
    grey = image.clone();

  tracker->detectorCascade->imgWidth = grey.cols;
  tracker->detectorCascade->imgHeight = grey.rows;
  tracker->detectorCascade->imgWidthStep = grey.step;

  tracker->selectObject(grey, videoInfo->initRoi);

  videoInfo->cur = videoInfo->cur->next;


  Rect bb;
  for ( ;; ) {
    // get frame from the video
    0[videoInfo->video] >> image;
    // stop the program if no more images
    if(image.rows==0 || image.cols==0)
      break;

    tracker->processImage(image);

    currOverlap = 0.;
    if(tracker->currBB != NULL)
    {
      bb = Rect(tracker->currBB->x, tracker->currBB->y, tracker->currBB->width, tracker->currBB->height);
      rectangle(image, bb, Scalar( 0, 255, 0), 2, 1 );
      if (videoInfo->cur) {
        currOverlap = tld::tldOverlapRectRect(bb, videoInfo->cur->rect[0]);
        totalOverlap += currOverlap;
        if (currOverlap > 0.5)
          tP ++;
        else
          fP ++;
      }
      else
      {
        fP ++;
      }


    }
    else
    {
      if (videoInfo->cur)
        fN ++;
      else
        tN++;
    }

    if (videoInfo->cur) {
      rectangle(image, videoInfo->cur->rect[0], Scalar( 0, 0, 255), 2, 1 );
      videoInfo->cur = videoInfo->cur->next;
      numFrame++;
    }

    namedWindow("tracker", CV_WINDOW_AUTOSIZE);
    imshow("tracker", image);

    //quit on ESC button
    if(waitKey(20)==27)break;
  }

  float prec = tP / (tP + fP), rec = tP / (tP + fN);
  cout << "Precision: " << prec << endl;
  cout << "Recall: " << rec << endl;
  cout << "F: " << 2 * prec * rec / (prec + rec) << endl;
  cout << "Average overlap: " << totalOverlap / numFrame << endl;

  videoInfo->cur = videoInfo->head;

  while (videoInfo->cur) {
    RectList *temp = videoInfo->cur->next;
    delete videoInfo->cur->rect;
    delete videoInfo->cur;
    videoInfo->cur = temp;
  }

  //free memory of video info
  delete videoInfo->video;
  delete tracker;
  free(videoInfo);

  return 0;
}
