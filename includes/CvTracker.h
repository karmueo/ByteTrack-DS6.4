#pragma once

#include <opencv2/opencv.hpp>
#include "nvbufsurface.h"
#include "nvdstracker.h"

class CvTracker
{
public:
    CvTracker(int frame_rate = 30, int track_buffer = 30);

    ~CvTracker();

    void update(const NvMOTProcessParams *params);

private:
    void init();

private:
    cv::Ptr<cv::Tracker> tracker_;
    // the intermediate scratch buffer for conversions RGBA
    NvBufSurface *inter_buf_;
    cv::Mat *cvmat_;
};