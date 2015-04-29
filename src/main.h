
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/video/background_segm.hpp>
#include <vector>

using namespace std;
cv::Mat ref_var, ref_mean;

cv::Ptr<cv::BackgroundSubtractor> pMOG;
cv::Mat fgMask;

int imgWidth;
int imgHeight;

const static int reqMatches = 4;
const static int radius = 40;
const static int subsamplingFactor = 4;

bool needsInit = true;
std::vector<std::vector<std::vector<uchar>>> samples;

#if defined(WIN32) || defined(_WIN32) 
#define PATH_SEPARATOR "\\" 
#else 
#define PATH_SEPARATOR "/" 
#endif

int processVideo(string path, string filename, int frames, int initframes);

cv::Mat processFrame(cv::Mat image, int learnframes, int framenum);

cv::Mat processFrame2(cv::Mat image, int learnframes, int framenum);

cv::Mat processFrameWithMOG(cv::Mat image, int learnframes, int framenum);

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

void printMat(cv::Mat mat)
{
	string ty = type2str(mat.type());
	printf("Matrix: %s %dx%d \n", ty.c_str(), mat.cols, mat.rows);
}