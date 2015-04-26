
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
void processFrame(cv::Mat image, int learnframes, int framenum);
