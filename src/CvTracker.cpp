#include "CvTracker.h"

CvTracker::CvTracker(int frame_rate, int track_buffer)
{
}

CvTracker::~CvTracker()
{
    delete cvmat_;
}

void CvTracker::init()
{
    /* Memset the memory */
    NvBufSurfaceMemSet(inter_buf_, 0, 0, 0);
}

void CvTracker::update(const NvMOTProcessParams *params)
{
    for (uint32_t i = 0; i < params->numFrames; i++)
    {
        NvMOTFrame *frame = &params->frameList[i];
        NvBufSurfaceParams *in_surf_params = frame->bufferList[0];
        cv::Mat in_mat = cv::Mat(
            in_surf_params->height,
            in_surf_params->width,
            CV_8UC4,
            in_surf_params->mappedAddr.addr[0],
            in_surf_params->pitch);

        cv::cvtColor(in_mat, *cvmat_, cv::COLOR_RGBA2BGR);
        cv::imwrite("out_" + std::to_string(i) + ".jpeg", *cvmat_);
    }
}
