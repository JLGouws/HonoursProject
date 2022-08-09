#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <cstring>
#include <iostream>
#include <stdlib.h>

using namespace std;
using namespace cv;

typedef struct RectList {
  RectList *next;
  Rect *rect;
} RectList;

typedef struct VidInfo {
  VideoCapture *video;
  Rect *initRoi;
  RectList *head;
  RectList *cur;
} VidInfo;

VidInfo* load_video_info(string basePath, string video) {
  VidInfo *videoInfo = (VidInfo *) malloc(sizeof(VidInfo));
  string suffix = "";
  string roiPath = basePath + video + "groundtruth_rect" + suffix + ".txt";
 // cout << roiPath << endl;
  FILE *file;
  file = fopen(const_cast<char*>(roiPath.c_str()), "r");
  if(file)
  {
    int value = 0;
    int state = 0, frame = 0;
    int values[4];
    char c = EOF;
    while((c = getc(file)) != EOF)
    {
      switch(c) 
      {
        case ',':
          if (state < 4)
          {
            state[values] = value;
            value  = 0;
            state += 1;
          }
          break;
        case '\n':
          state[values] = value;
          value  = 0;
          state = 0;
          if (frame == 0){
            videoInfo->initRoi = new Rect(0[values], 1[values], 2[values], 3[values]);
            videoInfo->head = (RectList*) malloc(sizeof(RectList));
            videoInfo->cur = videoInfo->head;
            videoInfo->head->rect = videoInfo->initRoi;
          }
          else {
            videoInfo->cur->next = (RectList*) malloc(sizeof(RectList));
            videoInfo->cur = videoInfo->cur->next;
            videoInfo->cur->rect = new Rect(0[values], 1[values], 2[values], 3[values]);
          }
          frame++;
          break;
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
          value = 10 * value + (c - '0');
          break;
      }
    }
  videoInfo->cur->next = NULL;
  videoInfo->cur = videoInfo->head;

  fclose(file);
  } else {
    cerr << "Could not find ROI file: " << roiPath << endl;
  }
  string videoPath = basePath + video + "img/"; 
  videoInfo->video = new VideoCapture(videoPath + "%04d.jpg");;
  return videoInfo;
}
