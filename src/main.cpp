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
	image = cv::imread(fullpath, CV_LOAD_IMAGE_COLOR);
	//image.convertTo(image, CV_32FC3);
	ref_mean = cv::Mat::zeros(image.rows, image.cols,CV_32FC3);
	ref_var = cv::Mat::zeros(image.rows, image.cols,CV_32FC3);
	// read files in loop
	for (int i = 0; i < frames; i++)
	{
		ostringstream ss;
		ss << setw(4) << setfill('0') << i;
		string str_i(ss.str());
		string fullpath = path + PATH_SEPARATOR + filename + "_" + str_i +".jpeg";
		image = cv::imread(fullpath, CV_LOAD_IMAGE_COLOR);
		if (!image.data)
		{
			printf(" No image data \n ");
		}
		cv::Mat result = processFrame(image, initframes,i);
		string savepath = path + PATH_SEPARATOR + "Seg_" + filename + "_" + str_i + ".jpeg";
		if (result.data && !DEBUG)
		{
			cv::imwrite(savepath, result);
		}
		string s = "Processing frame " + fullpath+"\n";
		printf(s.c_str());
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
		add(ref_mean, delta*(1/frame), ref_mean);
		subtract(image, ref_mean, tmp);
		add(ref_var, delta.mul(tmp), ref_var);
		if (framenum == learnframes - 1)
		{
			ref_var.mul(learnframes - 1);
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
		tmp.convertTo(tmp, image.type());
		tmp = delta >= tmp;
		cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
		cv::namedWindow("haha", cv::WINDOW_AUTOSIZE);
		cv::imshow("haha", tmp);
		//cv::GaussianBlur(tmp, tmp, cv::Size(3, 3), 0, 0);
		cv::threshold(tmp, tmp, 200.0, 255, cv::THRESH_BINARY);
		cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
		//cv::morphologyEx(tmp, tmp, cv::MORPH_OPEN, element);
		// adapt model
		// calculate coefficient alpha per pixel
		// in form of: alpha for 0 (background) and 1-alpha for 1 (foreground)
		float alpha = 0.1f;
		tmp.copyTo(delta);
		delta.convertTo(delta, CV_32F);
		delta = delta + (1-alpha);
		cv::threshold(delta, delta,max(alpha,1-alpha),  1 - alpha, cv::THRESH_TRUNC);

		cv::cvtColor(delta,delta, CV_GRAY2BGR,3);
		cv::Mat reciAlpha, change;
		cv::subtract(cv::Scalar::all(1.0f), delta, reciAlpha);
		cv::absdiff(image, ref_mean, change);
		ref_var = ref_var.mul(delta) + delta.mul(change);
		ref_mean = ref_mean.mul(delta) + image.mul(reciAlpha);
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
			cv::waitKey();
		}
		tmp.convertTo(tmp, CV_8UC3);
		return tmp;
	}
}