#include "main.h"
#include <cstdint>
#include <math.h>
#define DEBUG 1
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
	image = cv::imread(fullpath, cv::IMREAD_COLOR);
	ref_mean = cv::Mat::zeros(image.rows, image.cols,CV_32FC3);
	ref_var = cv::Mat::zeros(image.rows, image.cols,CV_32FC3);
	// read files in loop
	for (int i = 0; i < frames; i++)
	{
        string s = "Loading frame " + fullpath+"\n";
		printf(s.c_str());
    
		ostringstream ss;
		ss << setw(4) << setfill('0') << i;
		string str_i(ss.str());
		string fullpath = path + PATH_SEPARATOR + filename + "_" + str_i +".jpeg";

		image = cv::imread(fullpath, cv::IMREAD_COLOR);
		if (!image.data)
		{
			printf(" No image data \n ");
		}
      
        s = "Processing frame " + fullpath+"\n";
		printf(s.c_str());
		cv::Mat result = processFrame(image, initframes,i);
      
		string savepath = path + PATH_SEPARATOR + "Seg_" + filename + "_" + str_i + ".jpeg";
		if (result.data && !DEBUG)
		{
			cv::imwrite(savepath, result);
		}
		
	}
	return 0;
}
cv::Mat processFrame(cv::Mat image, int learnframes, int framenum)
{	
	image.convertTo(image, CV_32FC3);
	cv::Mat delta = cv::Mat::zeros(image.size(), image.type());
	cv::Mat tmp = cv::Mat::zeros(image.size(), image.type());
	// learn background
	if (framenum < learnframes)
	{
	// online calculation http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
		float frame = framenum + 1.0f;
		subtract(image, ref_mean, delta);
		add(ref_mean, delta.mul(1.0f/frame), ref_mean);
		subtract(image, ref_mean, tmp);
		add(ref_var, delta.mul(tmp), ref_var);
		if (framenum == learnframes - 1)
		{
			ref_var = ref_var.mul(1.0f/(learnframes - 1));
			sqrt(ref_var, tmp);
			ref_var = tmp;
		}
		return cv::Mat();
	}
	else
	{
		// segment foreground 
		cv::absdiff(image, ref_mean, delta);
		// constant = how many sigmas distance
		tmp = ref_var*3.5f;
		tmp = delta >= tmp;
		cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
		cv::GaussianBlur(tmp, tmp, cv::Size(3, 3), 0, 0);
		cv::threshold(tmp, tmp, 200.0, 255, cv::THRESH_BINARY);
		cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		cv::morphologyEx(tmp, tmp, cv::MORPH_OPEN, element);

		// adapt model
		// calculate coefficient alpha per pixel
		// in form of: alpha for 0 (background) and 1-alpha for 1 (foreground)
		float fgAlpha = 0.1f;
		float bgAlpha = 1.0f-fgAlpha;
		tmp.copyTo(delta);
		delta.convertTo(delta, CV_32F);
		cv::Mat mask = delta == 0;
		delta.setTo(bgAlpha, mask);
		mask = delta >= 0.99999f;
		delta.setTo(fgAlpha, mask);

		cv::cvtColor(delta,delta, cv::COLOR_GRAY2BGR,3);
		cv::Mat reciAlpha, change;
		cv::subtract(cv::Scalar::all(1.0f), delta, reciAlpha);
		cv::namedWindow("haha", cv::WINDOW_AUTOSIZE);
		cv::imshow("haha", reciAlpha);
		cv::absdiff(image, ref_mean, change);
		ref_var = ref_var.mul(reciAlpha) + delta.mul(change);
		ref_mean = ref_mean.mul(reciAlpha) + image.mul(delta);
		if (DEBUG)
		{
			cv::Mat tmpMean, tmpVar, tmpTmp, tmpImage;
			tmp.convertTo(tmpTmp, CV_8UC3);
			ref_mean.convertTo(tmpMean, CV_8UC3);
			ref_var.convertTo(tmpVar, CV_8UC3);
			image.convertTo(tmpImage, CV_8UC3);
			cv::namedWindow("Mean", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Var", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Seg", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Org", cv::WINDOW_AUTOSIZE);
			cv::imshow("Mean", tmpMean);
			cv::imshow("Var", tmpVar);
			cv::imshow("Seg", tmpTmp);
			cv::imshow("Org", tmpImage);
			cv::waitKey(10);
		}
		tmp.convertTo(tmp, CV_8UC3);
		return tmp;
	}
}