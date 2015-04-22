#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main( int argc, char** argv )
{
 const char* imageName = std::string("/Users/markomlinaric/Desktop/test.png").c_str();

 cv::Mat image;
 image = cv::imread( imageName, 1 );

 if(!image.data )
 {
   printf( " No image data \n " );
   return -1;
 }

 cv::Mat gray_image;
 cv::cvtColor( image, gray_image, cv::COLOR_BGR2GRAY );

 //cv::imwrite( "../../images/Gray_Image.jpg", gray_image );

 cv::namedWindow( imageName, cv::WINDOW_AUTOSIZE );
 cv::namedWindow( "Gray image", cv::WINDOW_AUTOSIZE );

 cv::imshow( imageName, image );
 cv::imshow( "Gray image", gray_image );

 cv::waitKey(0);

 return 0;
}