#include "main.h"
#include <cstdint>
#include <math.h>

int main(int argc, char** argv)
{
	if (argc < 5)
	{
		cout << "Not enough arguments" << endl;
		return 1;
	}
	string path, filename;
	int frames, initframes;
	path = argv[1];
	filename = argv[2];
	frames = atoi(argv[3]);
	initframes = atoi(argv[3]);
	processVideo(path, filename, frames, initframes);
	return 0;
}

int processVideo(string path, string filename, int frames, int initframes)
{
	cv::Mat image;
	// read files in loop
	for (int i = 0; i < frames; i++)
	{
		ostringstream ss;
		ss << setw(4) << setfill('0') << i;
		string str_i(ss.str());
		string fullpath = path + PATH_SEPARATOR + filename + "_" + str_i;
		image = cv::imread(fullpath, CV_LOAD_IMAGE_COLOR);
		processFrame(image, i < initframes);
		string s = "Processing frame " + fullpath+"\n";
		printf(s.c_str());
	}
}
void processFrame(cv::Mat image, bool isLearnFrame)
{
	string s = std::to_string(image.channels());
	printf(s.c_str());
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			cv::Vec3b bgrPixel = image.at<cv::Vec3b>(i, j);

			// do something with BGR values...
		}
	}
	// convert to grayscale
	//cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
	//cv::imwrite( "../../images/Gray_Image.jpg", gray_image );
	//cv::namedWindow(imageName, cv::WINDOW_AUTOSIZE);
	//cv::namedWindow("Gray image", cv::WINDOW_AUTOSIZE);
	//cv::imshow(imageName, image);
}