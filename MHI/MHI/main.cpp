#include "cv.h"
#include "highgui.h"
#include "stdlib.h"
#include "malloc.h"
#include "cxcore.h"
#include "assert.h"
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>




using namespace std;
using namespace cv;



// various tracking parameters (in seconds)
const double MHI_DURATION = 1;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;
// number of cyclic frame buffer used for motion detection
// (should, probably, depend on FPS)
//用于运动检测的循环帧数
const int N = 4;

// ring image buffer
IplImage **buf = 0;
int last = 0;

// temporary images
IplImage *mhi = 0; // MHI
IplImage *orient = 0; // orientation
IplImage *mask = 0; // valid orientation mask
IplImage *segmask = 0; // motion segmentation map
CvMemStorage* storage = 0; // temporary storage

// parameters:
//  img - input video frame
//  dst - resultant motion picture
//  args - optional parameters
void  update_mhi(IplImage* img, IplImage* dst, int diff_threshold)
{
	double timestamp = (double)clock() / CLOCKS_PER_SEC; // get current time in seconds
	CvSize size = cvSize(img->width, img->height); // get current frame size
	int i, idx1 = last, idx2;
	IplImage* silh;
	CvSeq* seq;
	CvRect comp_rect;
	double count;
	double angle;
	CvPoint center;
	double magnitude;
	CvScalar color;

	// allocate images at the beginning or  为图像分配初始空间
	// reallocate them if the frame size is changed  当帧的大小改变时，重新分配内存空间
	if (!mhi || mhi->width != size.width || mhi->height != size.height)
	{
		if (buf == 0)
		{
			buf = (IplImage**)malloc(N*sizeof(buf[0]));
			memset(buf, 0, N*sizeof(buf[0]));//把申请到的内存空间用0初始化
		}

		for (i = 0; i < N; i++) {
			cvReleaseImage(&buf[i]);
			buf[i] = cvCreateImage(size, IPL_DEPTH_8U, 1);
			cvZero(buf[i]);
		}
		cvReleaseImage(&mhi);
		cvReleaseImage(&orient);
		cvReleaseImage(&segmask);
		cvReleaseImage(&mask);

		mhi = cvCreateImage(size, IPL_DEPTH_32F, 1);
		cvZero(mhi); // clear MHI at the beginning
		orient = cvCreateImage(size, IPL_DEPTH_32F, 1);
		segmask = cvCreateImage(size, IPL_DEPTH_32F, 1);
		mask = cvCreateImage(size, IPL_DEPTH_8U, 1);
	}

	cvCvtColor(img, buf[last], CV_BGR2GRAY); // convert frame to grayscale 转换为灰度图

	idx2 = (last + 1) % N; // index of (last - (N-1))th frame
	last = idx2;
	silh = buf[idx2];
	cvAbsDiff(buf[idx1], buf[idx2], silh); // get difference between frames相邻两帧之差

	cvThreshold(silh, silh, diff_threshold, 1, CV_THRESH_BINARY); // and threshold it
	cvUpdateMotionHistory(silh, mhi, timestamp, MHI_DURATION); // update MHI  

	// convert MHI to blue 8u image
	cvCvtScale(mhi, mask, 255 / MHI_DURATION,
		(MHI_DURATION - timestamp) * 255 / MHI_DURATION);
	cvZero(dst);
	cvCvtPlaneToPix(mask, 0, 0, 0, dst);

	// calculate motion gradient orientation and valid orientation mask计算运动的梯度方向以及正确的方向掩码
	cvCalcMotionGradient(mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3);

	if (!storage)
		storage = cvCreateMemStorage(0);
	else
		cvClearMemStorage(storage);

	// segment motion: get sequence of motion components运动分割:获取运动组成成分的序列
	// segmask is marked motion components map. It is not used further
	seq = cvSegmentMotion(mhi, segmask, storage, timestamp, MAX_TIME_DELTA);

	// iterate through the motion components,迭代运动的组成成分
	// One more iteration (i == -1) corresponds to the whole image (global motion)
	for (i = -1; i < seq->total; i++)
	{

		if (i < 0)
		{ // case of the whole image
			comp_rect = cvRect(0, 0, size.width, size.height);
			color = CV_RGB(255, 255, 255);
			magnitude = 250;
		}
		else
		{ // i-th motion component
			comp_rect = ((CvConnectedComp*)cvGetSeqElem(seq, i))->rect;
			if (comp_rect.width + comp_rect.height < 100) // reject very small components
				continue;
			color = CV_RGB(255, 0, 0);
			magnitude = 30;
		}

		// select component ROI
		cvSetImageROI(silh, comp_rect);
		cvSetImageROI(mhi, comp_rect);
		cvSetImageROI(orient, comp_rect);
		cvSetImageROI(mask, comp_rect);

		// calculate orientation在选择区域内计算运动方向
		angle = cvCalcGlobalOrientation(orient, mask, mhi, timestamp, MHI_DURATION);
		angle = 360.0 - angle;  // adjust for images with top-left origin

		count = cvNorm(silh, 0, CV_L1, 0); // calculate number of points within silhouette ROI

		cvResetImageROI(mhi);
		cvResetImageROI(orient);
		cvResetImageROI(mask);
		cvResetImageROI(silh);

		// check for the case of little motion
		if (count < comp_rect.width*comp_rect.height * 0.05)
			continue;


	}
}


int main(int argc, char** argv)
{
	IplImage* motion = 0;
	CvCapture* capture = 0;

	capture = cvCaptureFromFile("F:\\KTH\\walking\\100.avi");
	//VideoCapture capture("walking.avi");
	if (capture)
	{
		cvNamedWindow("Motion", 1);

		for (;;)
		{
			IplImage* image;
			if (!cvGrabFrame(capture))
				break;
			//Decodes and returns the grabbed video frame.
			//C++: bool VideoCapture::retrieve(Mat& image, int channel=0)
			// IplImage* cvRetrieveFrame(CvCapture* capture, int streamIdx=0 )
			image = cvRetrieveFrame(capture);

			if (image)
			{
				if (!motion)
				{
					//Creates an image header and allocates the image data.
					//C: IplImage* cvCreateImage(CvSize size, int depth, int channels)
					motion = cvCreateImage(cvSize(image->width, image->height), 8, 3);
					//Clears the array.
					//C: void cvSetZero(CvArr* arr)
					cvZero(motion);
					motion->origin = image->origin;
				}
			}

			update_mhi(image, motion, 60);
			cvShowImage("Motion", motion);

			if (cvWaitKey(30) >= 0)
				break;
		}
		cvReleaseCapture(&capture);
		cvDestroyWindow("Motion");
	}

	return 0;

}