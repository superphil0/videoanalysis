#include "main.h"
#include <cstdint>
#include <math.h>
#define DEBUG 0
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
    
    //pMOG = cv::createBackgroundSubtractorMOG2();
    
	// read files in loop
	for (int i = 0; i < frames; i++)
	{
		ostringstream ss;
		ss << setw(4) << setfill('0') << i;
		string str_i(ss.str());
		string fullpath = path + PATH_SEPARATOR + filename + "_" + str_i +".jpeg";

        string s = "Loading frame " + fullpath+"\n";
		printf("%s", s.c_str());

		image = cv::imread(fullpath, cv::IMREAD_COLOR);
		if (!image.data)
		{
			printf(" No image data \n ");
            return 0;
		}
      
        s = "Processing frame " + fullpath+"\n";
		printf("%s", s.c_str());
		cv::Mat result = processFrameVIBE(image, initframes, i);
      
		string savepath = path + PATH_SEPARATOR + "Seg_" + filename + "_" + str_i + ".jpeg";
		if (result.data && !DEBUG)
		{
			cv::imwrite(savepath, result);
		}
		
	}
	return 0;
}
cv::Mat processFrameCMV(cv::Mat image, int learnframes, int framenum)
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
			//sqrt(ref_var, tmp);
			ref_var = tmp;
		}
		return cv::Mat();
	}
	else
	{
		// segment foreground 
		cv::absdiff(image, ref_mean, delta);
		// constant = how many sigmas distance
		sqrt(ref_var, tmp);
		tmp = tmp * 3.5f;
		tmp = delta >= tmp;
		cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
		cv::GaussianBlur(tmp, tmp, cv::Size(3, 3), 0, 0);
		cv::threshold(tmp, tmp, 200.0, 255, cv::THRESH_BINARY);
		cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		cv::morphologyEx(tmp, tmp, cv::MORPH_CLOSE, element);
		vector<vector<cv::Point>> contours;
		cv::findContours(tmp, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
		
		for( int i = 0; i< contours.size(); i++ )
		 {
		   cv::drawContours(tmp, contours, i, 255, -1);
		 }
    

		// adapt model
		// calculate coefficient alpha per pixel
		// in form of: alpha for 0 (background) and 1-alpha for 1 (foreground)
		float fgAlpha = 0.05f;
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
		ref_var = ref_var.mul(reciAlpha) + delta.mul(change.mul(change));
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

void chooseRandomNeighbor(int x, int y, int &xn, int &yn) {
    xn = x + rand()%5 - 2;
    yn = y + rand()%5 - 2;
    
    if (xn < 0) xn = 0;
    if (xn > imgWidth-1) xn = imgWidth-1;
    if (yn < 0) yn = 0;
    if (yn > imgHeight-1) yn = imgHeight-1;
}

cv::Mat processFrameVIBE(cv::Mat image, int learnframes, int framenum) {
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    // we need to initialize the data structure before the first frame
    if (needsInit) {
        if (learnframes < nbSamples) {
            nbSamples = learnframes;
        }
    
        cv::Size s = image.size();
        imgWidth = s.width;
        imgHeight = s.height;
    
        samples.resize(imgWidth);
        for (int i = 0; i < imgWidth; i++) {
            samples[i].resize(imgHeight);
            for (int j = 0; j < imgHeight; j++) {
                samples[i][j].resize(nbSamples);
            }
        }
        
        needsInit = false;
    }
    
    // let's do some initial sampling (NOTE: we do not sample from the neighbor like it was mentioned in the paper)
    if (framenum < learnframes - nbSamples) {
        return cv::Mat();
    }
    if (framenum < learnframes)
	{
        for (int x = 0; x < imgWidth; x++){
            for (int y = 0; y < imgHeight; y++){
                samples[x][y][framenum-learnframes+nbSamples] = image.at<uchar>(cv::Point(x,y));
            }
        }
        return cv::Mat();
	}
    
    // this holds our result
    cv::Mat segmentationMap = cv::Mat::zeros(image.size(), CV_8U);
    
    // TODO this could be done in a OpenCV way using Matrixes
    for (int x = 0; x < imgWidth; x++){
        for (int y = 0; y < imgHeight; y++){
        
            // we look for as many matches as defined by reqMatches
            int count = 0;
            int index = 0;
            while ((count < reqMatches) && (index < nbSamples)){
                uchar color_a = image.at<uchar>(cv::Point(x,y));
                uchar color_b = samples[x][y][index];
                int distance = abs(color_a - color_b);
                //distance = cv::norm(color_a, color_b, cv::NORM_L2);
                if (distance < radius)
                    count++;
                index++;
            }
            
            // if we have enough matches, the pixel belongs to the background
            if (count >= reqMatches) {
                segmentationMap.at<uchar>(cv::Point(x,y)) = 0;
                
                // we might now update a sample for the current pixel
                int randomNumber = rand()%subsamplingFactor;
                if (randomNumber == 0) {
                    randomNumber = rand()%nbSamples;
                    samples[x][y][randomNumber] = image.at<uchar>(cv::Point(x,y));
                }
                
                // we might now update a neighbor pixel
                randomNumber = rand()%subsamplingFactor;
                if (randomNumber == 0) {
                    int neighborX, neighborY;
                    chooseRandomNeighbor(x, y, neighborX, neighborY);
                    randomNumber = rand()%nbSamples;
                    samples[neighborX][neighborY][randomNumber] = image.at<uchar>(cv::Point(x,y));
                }	
            }
            else
            {
                // pixel in foreground
                segmentationMap.at<uchar>(cv::Point(x,y)) = 255;
            }
        }
    }
    
    // here we do some postprocessing
    
    // closing removes some noise and helps to close gaps between components
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(segmentationMap, segmentationMap, cv::MORPH_CLOSE, element);
    
    // we fill contours, as we assume that objects don't have any holes
    vector<vector<cv::Point>> contours;
    cv::findContours(segmentationMap, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    for( int i = 0; i< contours.size(); i++ )
    {
       cv::drawContours(segmentationMap, contours, i, 255, -1);
    }
    
    //cv::medianBlur(segmentationMap, segmentationMap, 3);
    
    if (DEBUG) {
        cv::imshow("Org", image);
        cv::imshow("Seg", segmentationMap);
        cv::waitKey();
    }
    
    return segmentationMap;
}

cv::Mat processFrameWithMOG(cv::Mat image, int learnframes, int framenum) {
//    pMOG->apply(image, fgMask);
    if (DEBUG) {
        cv::imshow("Org", image);
        cv::imshow("Seg", fgMask);
        cv::waitKey();
    }
    return image;
}