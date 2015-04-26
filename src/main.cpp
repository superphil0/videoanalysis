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
	initframes = atoi(argv[4]);
	processVideo(path, filename, frames, initframes);
	return 0;
}

int processVideo(string path, string filename, int frames, int initframes)
{
	cv::Mat image;
	string fullpath = path + PATH_SEPARATOR + filename + "_0000.jpeg";
	image = cv::imread(fullpath, CV_LOAD_IMAGE_COLOR);
	//image.convertTo(image, CV_32FC3);
	ref_mean = cv::Mat::zeros(image.rows, image.cols,image.type());
	ref_var = cv::Mat::zeros(image.rows, image.cols,image.type());
	// read files in loop
	for (int i = 0; i < frames; i++)
	{
		ostringstream ss;
		ss << setw(4) << setfill('0') << i;
		string str_i(ss.str());
		string fullpath = path + PATH_SEPARATOR + filename + "_" + str_i +".jpeg";
		image = cv::imread(fullpath, CV_LOAD_IMAGE_COLOR);
		string s1 = to_string(image.channels()) +"  :  "+ to_string(image.type());
		printf(s1.c_str());
		if (!image.data)
		{
			printf(" No image data \n ");
		}
		processFrame(image, initframes,i);
		string s = "Processing frame " + fullpath+"\n";
		printf(s.c_str());
	}
	return 0;
}
void processFrame(cv::Mat image, int learnframes, int framenum)
{	
	// online calculation http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
	if (framenum < learnframes)
	{
		float frame = framenum + 1.0f;
		cv::Mat delta = cv::Mat::zeros(image.size(), image.type());
		cv::Mat tmp = cv::Mat::zeros(image.size(), image.type());
		subtract(image, ref_mean, delta);
		add(ref_mean, delta*(1/frame), ref_mean);
		subtract(image, ref_mean, tmp);
		add(ref_var, delta.mul(tmp), ref_var);
		if (framenum == learnframes - 1)
		{
			ref_var.mul(learnframes - 1);
		}
	}
	else
	{
		cv::namedWindow("Mean", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("Var", cv::WINDOW_AUTOSIZE);
		cv::imshow("Mean", ref_mean);
		cv::imshow("Var", ref_var);
		cv::waitKey();
	}
	// convert to grayscale
	//cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
	//cv::imwrite( "../../images/Gray_Image.jpg", gray_image );
	//cv::namedWindow(imageName, cv::WINDOW_AUTOSIZE);
	//cv::namedWindow("Gray image", cv::WINDOW_AUTOSIZE);
	//cv::imshow(imageName, image);
}