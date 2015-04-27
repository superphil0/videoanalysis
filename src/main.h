
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>
using namespace std;
cv::Mat ref_var, ref_mean;	
#if defined(WIN32) || defined(_WIN32) 
#define PATH_SEPARATOR "\\" 
#else 
#define PATH_SEPARATOR "/" 
#endif

int processVideo(string path, string filename, int frames, int initframes);
cv::Mat processFrame(cv::Mat image, int learnframes, int framenum);
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