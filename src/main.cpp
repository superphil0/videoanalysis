#include "main.h"
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
		image = cv::imread(fullpath, 1);
		processFrame(image);
		string s = "Processing frame " + str_i+"\n";
		printf(s.c_str());
	}
}
void processFrame(cv::Mat image)
{
	cv::Mat gray_image;
	// convert to grayscale
	//cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
	//cv::imwrite( "../../images/Gray_Image.jpg", gray_image );
	//cv::namedWindow(imageName, cv::WINDOW_AUTOSIZE);
	//cv::namedWindow("Gray image", cv::WINDOW_AUTOSIZE);
	//cv::imshow(imageName, image);
}