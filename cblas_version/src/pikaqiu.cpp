#include "network.h"
#include "mtcnn.h"
#include <time.h>
int main(int argc, char** argv)
{
    const char* imagepath = argv[1];

    cv::Mat cv_img = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (cv_img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

	//Mat image = imread("/data/local/tmp/woman.jpg");
	mtcnn find(cv_img.rows, cv_img.cols);
    //int rows = image.rows;
	//int cols = image.cols;
	//print("rows:%d",rows);
	int64 starttime, endtime;
	double RUNtime = 0;
	unsigned int iteration = 500;
	starttime = cvGetTickCount();
	for (unsigned int i = 0; i < iteration; i++)
	{
		find.findFace(cv_img);
	}
	endtime = cvGetTickCount();
	//imshow("result", image);
	imwrite("/data/local/tmp/result.jpg", cv_img);
	RUNtime = (endtime - starttime) / (1000 * cvGetTickFrequency()*iteration);
	printf("Average RUN time:%f ms\n", RUNtime);
    //waitKey(0);
    cv_img.release();
    return 0;
}