#include "Tracker.h"
#include <fstream>
#include <cuda_runtime.h>

NvMOTContext::NvMOTContext(const NvMOTConfig &configIn, NvMOTConfigResponse &configResponse)
{
    configResponse.summaryStatus = NvMOTConfigStatus_OK;

    bool HOG = true;
    bool FIXEDWINDOW = false;
    bool MULTISCALE = true;
    bool LAB = false;

    // fdsstTracker = std::make_shared<FDSSTTracker>(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    fdsstTracker = std::make_shared<FDSSTTracker>();
    // cv_tracker_ = cv::TrackerKCF::create();
}

NvMOTContext::~NvMOTContext()
{
}

NvMOTStatus NvMOTContext::processFrame(const NvMOTProcessParams *params, NvMOTTrackedObjBatch *pTrackedObjectsBatch)
{
    cv::Mat in_mat;

    if (!params || params->numFrames <= 0)
    {
        return NvMOTStatus_OK;
    }

    for (uint streamIdx = 0; streamIdx < pTrackedObjectsBatch->numFilled; streamIdx++)
    {
        NvMOTTrackedObjList *trackedObjList = &pTrackedObjectsBatch->list[streamIdx];
        NvMOTFrame *frame = &params->frameList[streamIdx];

        if (frame->bufferList[0] == nullptr)
        {
            std::cout << "frame->bufferList[0] is nullptr" << std::endl;
            continue;
        }

        NvBufSurfaceParams *bufferParams = frame->bufferList[0];
        cv::Mat bgraFrame(bufferParams->height, bufferParams->width, CV_8UC4, bufferParams->dataPtr);
        // 转化为灰度图
        cv::cvtColor(bgraFrame, in_mat, cv::COLOR_BGRA2GRAY);
        if (is_tracked_ == false)
        {
            if (frame->objectsIn.numFilled == 0)
            {
                continue;
            }

            // cv::imwrite("out.jpeg", bgraFrame);

            // FIXME: 目前只跟踪第一个
            objectToTrack_ = &frame->objectsIn.list[0];
            Rect initRect = cv::Rect(objectToTrack_->bbox.x,
                                     objectToTrack_->bbox.y,
                                     objectToTrack_->bbox.width,
                                     objectToTrack_->bbox.height);

            // cv_tracker_->init(in_mat, initRect);

            fdsstTracker->init(initRect, in_mat);
            showRect_ = initRect;
            is_tracked_ = true;
        }
        else
        {
            showRect_ = fdsstTracker->update(in_mat);
            // cv::imwrite("in_mat.jpeg", in_mat);
            // cv_tracker_->update(in_mat, showRect_);
            // 把矩形框画在图像上
            // cv::rectangle(in_mat, showRect_, cv::Scalar(0, 255, 0), 2);
            // cv::imwrite("out.jpeg", in_mat);
        }

        if (trackedObjList->numAllocated != MAX_TARGETS_PER_STREAM)
        {
            // Reallocate memory space
            delete trackedObjList->list;
            trackedObjList->list = new NvMOTTrackedObj[MAX_TARGETS_PER_STREAM];
        }
        // This should resolve the memory leak issue:
        //   https://github.com/ifzhang/ByteTrack/issues/276
        NvMOTTrackedObj *trackedObjs = trackedObjList->list;
        int filled = 0;

        NvMOTRect motRect{showRect_.x, showRect_.y, showRect_.width, showRect_.height};
        NvMOTTrackedObj *trackedObj = &trackedObjs[filled];
        trackedObj->classId = 0;
        trackedObj->trackingId = 0;
        trackedObj->bbox = motRect;
        trackedObj->confidence = 1;
        trackedObj->age = age_++;
        trackedObj->associatedObjectIn = objectToTrack_;
        trackedObj->associatedObjectIn->doTracking = true;
        filled++;

        trackedObjList->streamID = frame->streamID;
        trackedObjList->frameNum = frame->frameNum;
        trackedObjList->valid = true;
        trackedObjList->list = trackedObjs;
        trackedObjList->numFilled = filled;
        trackedObjList->numAllocated = MAX_TARGETS_PER_STREAM;
    }
    return NvMOTStatus_OK;
}

NvMOTStatus NvMOTContext::retrieveMiscData(const NvMOTProcessParams *params,
                                           NvMOTTrackerMiscData *pTrackerMiscData)
{
    return NvMOTStatus_OK;
}

NvMOTStatus NvMOTContext::removeStream(const NvMOTStreamId streamIdMask)
{
    return NvMOTStatus_OK;
}
