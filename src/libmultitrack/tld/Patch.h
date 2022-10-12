#ifndef MYPATCH_H_
#define MYPATCH_H_

#include <opencv/cv.h>

namespace tld
{

class Patch
{
public:
    cv::Rect roi;
    bool positive;
};

} /* namespace tld */
#endif /* NORMALIZEDPATCH_H_ */
