#include <stdio.h>
#include <algorithm>
#include <vector>
#include <math.h>
#include <iostream>
#include <sys/time.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


#include "net.h"
#include"cpu.h"
using namespace std;
using namespace cv;

struct Bbox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool exist;
    float ppoint[10];
    float regreCoord[4];
};

struct orderScore
{
    float score;
    int oriOrder;
};
bool cmpScore(orderScore lsh, orderScore rsh){
    if(lsh.score<rsh.score)
        return true;
    else
        return false;
}
static float getElapse(struct timeval *tv1,struct timeval *tv2)
{
    float t = 0.0f;
    if (tv1->tv_sec == tv2->tv_sec)
        t = (tv2->tv_usec - tv1->tv_usec)/1000.0f;
    else
        t = ((tv2->tv_sec - tv1->tv_sec) * 1000 * 1000 + tv2->tv_usec - tv1->tv_usec)/1000.0f;
    return t;
}

void resize_image(ncnn::Mat& srcImage, ncnn::Mat& dstImage)
{
	int src_width = srcImage.w;
	int src_height = srcImage.h;
	int src_channel = srcImage.c;
	int dst_width = dstImage.w;
	int dst_height = dstImage.h;
	int dst_channel = dstImage.c;

	if (src_width==dst_width && src_height==dst_height)
	{
		memcpy(dstImage.data, srcImage.data, src_width*src_height*src_channel*sizeof(float));
		return;
	}
	float lf_x_scl = static_cast<float>(src_width) / dst_width;
	float lf_y_Scl = static_cast<float>(src_height) / dst_height;
	const float* src_data = srcImage.data;

	float* dest_data = dstImage.data;
	int src_area = srcImage.cstep;
	int src_area2 = 2 * src_area;
	int dst_area = dstImage.cstep;
	int dst_area2 = 2 * dst_area;

	for (int y = 0; y < dst_height; y++) {
		for (int x = 0; x < dst_width; x++) {
			float lf_x_s = lf_x_scl * x;
			float lf_y_s = lf_y_Scl * y;

			int n_x_s = static_cast<int>(lf_x_s);
			n_x_s = (n_x_s <= (src_width - 2) ? n_x_s : (src_width - 2));
			int n_y_s = static_cast<int>(lf_y_s);
			n_y_s = (n_y_s <= (src_height - 2) ? n_y_s : (src_height - 2));

			float lf_weight_x = lf_x_s - n_x_s;
			float lf_weight_y = lf_y_s - n_y_s;

			float dest_val_b = (1 - lf_weight_y) * ((1 - lf_weight_x) *
				src_data[n_y_s * src_width + n_x_s] +
				lf_weight_x * src_data[n_y_s * src_width + n_x_s + 1]) +
				lf_weight_y * ((1 - lf_weight_x) * src_data[(n_y_s + 1) * src_width + n_x_s] +
				lf_weight_x * src_data[(n_y_s + 1) * src_width + n_x_s + 1]);
			float dest_val_g = (1 - lf_weight_y) * ((1 - lf_weight_x) *
				src_data[n_y_s * src_width + n_x_s + src_area] +
				lf_weight_x * src_data[n_y_s * src_width + n_x_s + 1 + src_area]) +
				lf_weight_y * ((1 - lf_weight_x) * src_data[(n_y_s + 1) * src_width + n_x_s + src_area] +
				lf_weight_x * src_data[(n_y_s + 1) * src_width + n_x_s + 1 + src_area]);
			float dest_val_r = (1 - lf_weight_y) * ((1 - lf_weight_x) *
				src_data[n_y_s * src_width + n_x_s + src_area2] +
				lf_weight_x * src_data[n_y_s * src_width + n_x_s + 1 + src_area2]) +
				lf_weight_y * ((1 - lf_weight_x) * src_data[(n_y_s + 1) * src_width + n_x_s + src_area2] +
				lf_weight_x * src_data[(n_y_s + 1) * src_width + n_x_s + 1 + src_area2]);

			dest_data[y * dst_width + x] = static_cast<float>(dest_val_b);
			dest_data[y * dst_width + x + dst_area] = static_cast<float>(dest_val_g);
			dest_data[y * dst_width + x + 2 * dst_area] = static_cast <float>(dest_val_r);
		}
	}
} 

class mtcnn{
public:
    mtcnn();
    void detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox);
private:
    void generateBbox(ncnn::Mat score, ncnn::Mat location, vector<Bbox>& boundingBox_, vector<orderScore>& bboxScore_, float scale);
    void nms(vector<Bbox> &boundingBox_, std::vector<orderScore> &bboxScore_, const float overlap_threshold, string modelname="Union");
    void refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width);

    ncnn::Net Pnet, Rnet, Onet;
    ncnn::Mat img;

    const float nms_threshold[3] = {0.5, 0.6, 0.7};//{0.4, 0.4, 0.4};
    const float threshold[3] = {0.6, 0.7, 0.8};
    const float mean_vals[3] = {127.5, 127.5, 127.5};
    const float norm_vals[3] = {0.0078125, 0.0078125, 0.0078125};
    std::vector<Bbox> firstBbox_, secondBbox_,thirdBbox_;
    std::vector<orderScore> firstOrderScore_, secondBboxScore_, thirdBboxScore_;
    int img_w, img_h;
};

mtcnn::mtcnn(){
    Pnet.load_param("/data/local/tmp/det1.param");
    Pnet.load_model("/data/local/tmp/det1.bin");
    Rnet.load_param("/data/local/tmp/det2.param");
    Rnet.load_model("/data/local/tmp/det2.bin");
    Onet.load_param("/data/local/tmp/det3.param");
    Onet.load_model("/data/local/tmp/det3.bin");
}

/******************generateBbox******************************/
//根据Pnet的输出结果，由滑框的得分，筛选可能是人脸的滑框，并记录该框的位置、人脸坐标信息、得分以及编号
void mtcnn::generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox>& boundingBox_, std::vector<orderScore>& bboxScore_, float scale){
    int stride = 2;//Pnet中有一次MP2*2，后续转换的时候相当于stride=2；
    int cellsize = 12;
    int count = 0;
    //score p
    float *p = score.channel(1);//score.data + score.cstep;//判定为人脸的概率
    float *plocal = location.data;
    Bbox bbox;
    orderScore order;
    for(int row=0;row<score.h;row++){
        for(int col=0;col<score.w;col++){
            if(*p>threshold[0]){
                bbox.score = *p;//记录得分
                order.score = *p;
                order.oriOrder = count;//记录有效滑框的编号
                bbox.x1 = round((stride*col+1)/scale);//12*12的滑框，换算到原始图像上的坐标
                bbox.y1 = round((stride*row+1)/scale);
                bbox.x2 = round((stride*col+1+cellsize)/scale);
                bbox.y2 = round((stride*row+1+cellsize)/scale);
                bbox.exist = true;
                bbox.area = (bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
                for(int channel=0;channel<4;channel++)
                    bbox.regreCoord[channel]=location.channel(channel)[0];//人脸框的坐标相关值
                boundingBox_.push_back(bbox);
                bboxScore_.push_back(order);
                count++;
            }
            p++;
            plocal++;
        }
    }
}

/**********************nms非极大值抑制****************************/
void mtcnn::nms(std::vector<Bbox> &boundingBox_, std::vector<orderScore> &bboxScore_, const float overlap_threshold, string modelname){
    if(boundingBox_.empty()){
        return;
    }
    std::vector<int> heros;
    //sort the score
    sort(bboxScore_.begin(), bboxScore_.end(), cmpScore);//cmpScore指定升序排列

    int order = 0;
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
	//规则，站上擂台的擂台主，永远都是胜利者。
    while(bboxScore_.size()>0){
        order = bboxScore_.back().oriOrder;//取得分最高勇士的编号ID。
        bboxScore_.pop_back();//勇士出列
        if(order<0)continue;//死的？下一个！（order在(*it).oriOrder = -1;改变）
        heros.push_back(order);//记录擂台主ID
        boundingBox_.at(order).exist = false;//当前这个Bbox为擂台主，签订生死簿。

        for(int num=0;num<boundingBox_.size();num++){
            if(boundingBox_.at(num).exist){//活着的勇士
                //the iou
                maxX = (boundingBox_.at(num).x1>boundingBox_.at(order).x1)?boundingBox_.at(num).x1:boundingBox_.at(order).x1;
                maxY = (boundingBox_.at(num).y1>boundingBox_.at(order).y1)?boundingBox_.at(num).y1:boundingBox_.at(order).y1;
                minX = (boundingBox_.at(num).x2<boundingBox_.at(order).x2)?boundingBox_.at(num).x2:boundingBox_.at(order).x2;
                minY = (boundingBox_.at(num).y2<boundingBox_.at(order).y2)?boundingBox_.at(num).y2:boundingBox_.at(order).y2;
                //maxX1 and maxY1 reuse 
                maxX = ((minX-maxX+1)>0)?(minX-maxX+1):0;
                maxY = ((minY-maxY+1)>0)?(minY-maxY+1):0;
                //IOU reuse for the area of two bbox
                IOU = maxX * maxY;
                if(!modelname.compare("Union"))
                    IOU = IOU/(boundingBox_.at(num).area + boundingBox_.at(order).area - IOU);
                else if(!modelname.compare("Min")){
                    IOU = IOU/((boundingBox_.at(num).area<boundingBox_.at(order).area)?boundingBox_.at(num).area:boundingBox_.at(order).area);
                }
                if(IOU>overlap_threshold){
                    boundingBox_.at(num).exist=false;//如果该对比框与擂台主的IOU够大，挑战者勇士战死
                    for(vector<orderScore>::iterator it=bboxScore_.begin(); it!=bboxScore_.end();it++){
                        if((*it).oriOrder == num) {
                            (*it).oriOrder = -1;//勇士战死标志
                            break;
                        }
                    }
                }//else 那些距离擂台主比较远迎战者幸免于难，将有机会作为擂台主出现
            }
        }
    }
    for(int i=0;i<heros.size();i++)
        boundingBox_.at(heros.at(i)).exist = true;//从生死簿上剔除，擂台主活下来了
}
void mtcnn::refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width){
    if(vecBbox.empty()){
        cout<<"Bbox is empty!!"<<endl;
        return;
    }
    float bbw=0, bbh=0, maxSide=0;
    float h = 0, w = 0;
    float x1=0, y1=0, x2=0, y2=0;
    for(vector<Bbox>::iterator it=vecBbox.begin(); it!=vecBbox.end();it++){
        if((*it).exist){
            bbw = (*it).x2 - (*it).x1 + 1;//滑框的宽高计算
            bbh = (*it).y2 - (*it).y1 + 1;
            x1 = (*it).x1 + (*it).regreCoord[0]*bbw;//人脸框的位置坐标计算
            y1 = (*it).y1 + (*it).regreCoord[1]*bbh;
            x2 = (*it).x2 + (*it).regreCoord[2]*bbw;
            y2 = (*it).y2 + (*it).regreCoord[3]*bbh;

            w = x2 - x1 + 1;//人脸框宽高
            h = y2 - y1 + 1;
          
            maxSide = (h>w)?h:w;
            x1 = x1 + w*0.5 - maxSide*0.5;
            y1 = y1 + h*0.5 - maxSide*0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);

            //boundary check
            if((*it).x1<0)(*it).x1=0;
            if((*it).y1<0)(*it).y1=0;
            if((*it).x2>width)(*it).x2 = width - 1;
            if((*it).y2>height)(*it).y2 = height - 1;

            it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
        }
    }
}
void mtcnn::detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox_){
    img = img_;
    img_w = img.w;
    img_h = img.h;
    img.substract_mean_normalize(mean_vals, norm_vals);//数据预处理,归一化至(-1,1)

    float minl = img_w<img_h?img_w:img_h;
    int MIN_DET_SIZE = 12;
    int minsize = 240;//最小可检测图像，该值大小，控制图像金字塔的阶层数，越小，阶层越多，计算越多。
    float m = (float)MIN_DET_SIZE/minsize;
    minl *= m;
    float factor = 0.409;
    int factor_count = 0;
    vector<float> scales_;
    while(minl>MIN_DET_SIZE){
		if (factor_count > 0){ m = m*factor; }
        scales_.push_back(m);
        minl *= factor;
        factor_count++;
    }
    orderScore order;
    int count = 0;

    for (size_t i = 0; i < scales_.size(); i++) {
        int hs = (int)ceil(img_h*scales_[i]);
        int ws = (int)ceil(img_w*scales_[i]);
        //ncnn::Mat in = ncnn::Mat::from_pixels_resize(image_data, ncnn::Mat::PIXEL_RGB, img_w, img_h, ws, hs);
        ncnn::Mat in(ws, hs, 3);
        resize_image(img, in);//一次次生成图像金字塔中的一层图
        ncnn::Extractor ex = Pnet.create_extractor();
        ex.set_light_mode(true);
		//printf("Pnet input width:%d, height:%d, channel:%d\n",in.w,in.h,in.c);
        ex.input("data", in);//Pnet只有卷积层，所以可以接受不同size的input
        ncnn::Mat score_, location_;
        ex.extract("prob1", score_);
		//printf("prob1 w:%d, h:%d, ch:%d, first data:%f\n", score_.w, score_.h, score_.c, score_.data[0]);
		//for (int t_w = 0; t_w < score_.w*score_.h*score_.c; t_w++)
		//{
		//	printf("%f, ", score_.data[t_w]);
		//}
        ex.extract("conv4-2", location_);
        std::vector<Bbox> boundingBox_;
        std::vector<orderScore> bboxScore_;
        generateBbox(score_, location_, boundingBox_, bboxScore_, scales_[i]);
        nms(boundingBox_, bboxScore_, nms_threshold[0]);//分会场擂台赛

        for(vector<Bbox>::iterator it=boundingBox_.begin(); it!=boundingBox_.end();it++){
            if((*it).exist){//获胜擂台主得到进入主会场的机会
                firstBbox_.push_back(*it);//主会场花名册
                order.score = (*it).score;
                order.oriOrder = count;
                firstOrderScore_.push_back(order);
                count++;
            }
        }
        bboxScore_.clear();
        boundingBox_.clear();
    }
    //the first stage's nms
    if(count<1)return;
    nms(firstBbox_, firstOrderScore_, nms_threshold[0]);//主会场擂台赛
    refineAndSquareBbox(firstBbox_, img_h, img_w);
    //printf("firstBbox_.size()=%d\n", firstBbox_.size());
	//for (vector<Bbox>::iterator it = firstBbox_.begin(); it != firstBbox_.end(); it++)
	//{
	//	cout << "OK" << endl;
	//	//rectangle(cp_img, Point((*it).x1, (*it).y1), Point((*it).x2, (*it).y2), Scalar(0, 0, 255), 2, 8, 0);
	//}
	//imshow("Pnet.jpg", cp_img);
	//waitKey(1000);
    //second stage
    count = 0;
    for(vector<Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
        if((*it).exist){
            ncnn::Mat tempIm;
            copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
            ncnn::Mat in(24, 24, 3);
            resize_image(tempIm, in);
            ncnn::Extractor ex = Rnet.create_extractor();
            ex.set_light_mode(true);
            ex.input("data", in);
            ncnn::Mat score, bbox;
            ex.extract("prob1", score);
            ex.extract("conv5-2", bbox);
            if(*(score.data+score.cstep)>threshold[1]){
                for(int channel=0;channel<4;channel++)
                    it->regreCoord[channel]=bbox.channel(channel)[0];//*(bbox.data+channel*bbox.cstep);
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = score.channel(1)[0];//*(score.data+score.cstep);
                secondBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                secondBboxScore_.push_back(order);
            }
            else{
                (*it).exist=false;
            }
        }
    }
    //printf("secondBbox_.size()=%d\n", secondBbox_.size());
    if(count<1)return;
    nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
    refineAndSquareBbox(secondBbox_, img_h, img_w);

    //third stage 
    count = 0;
    for(vector<Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        if((*it).exist){
            ncnn::Mat tempIm;
            copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
            ncnn::Mat in(48, 48, 3);
            resize_image(tempIm, in);
            ncnn::Extractor ex = Onet.create_extractor();
            ex.set_light_mode(true);
            ex.input("data", in);
            ncnn::Mat score, bbox, keyPoint;
            ex.extract("prob1", score);
            ex.extract("conv6-2", bbox);
            ex.extract("conv6-3", keyPoint);
            if(score.channel(1)[0]>threshold[2]){
                for(int channel=0;channel<4;channel++)
                    it->regreCoord[channel]=bbox.channel(channel)[0];
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = score.channel(1)[0];
                for(int num=0;num<5;num++){
                    (it->ppoint)[num] = it->x1 + (it->x2 - it->x1)*keyPoint.channel(num)[0];
                    (it->ppoint)[num+5] = it->y1 + (it->y2 - it->y1)*keyPoint.channel(num+5)[0];
                }

                thirdBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                thirdBboxScore_.push_back(order);
            }
            else
                (*it).exist=false;
            }
        }

    //printf("thirdBbox_.size()=%d\n", thirdBbox_.size());
    if(count<1)return;
    refineAndSquareBbox(thirdBbox_, img_h, img_w);
    nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");
    finalBbox_ = thirdBbox_;

    firstBbox_.clear();
    firstOrderScore_.clear();
    secondBbox_.clear();
    secondBboxScore_.clear();
    thirdBbox_.clear();
    thirdBboxScore_.clear();
}

int main(int argc, char** argv)
{
    const char* imagepath = argv[1];

    cv::Mat cv_img = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (cv_img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    std::vector<Bbox> finalBbox;
    mtcnn mm;
    struct timeval  tv1,tv2;
    struct timezone tz1,tz2;
    struct timeval  tv3,tv4;
    struct timezone tz3,tz4;
	gettimeofday(&tv3,&tz3);
	unsigned int iter_time =500;
	for(unsigned int i=0;i<iter_time;i++)
		{	
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(cv_img.data, ncnn::Mat::PIXEL_BGR2RGB, cv_img.cols, cv_img.rows);	

    gettimeofday(&tv1,&tz1);
    mm.detect(ncnn_img, finalBbox);
    gettimeofday(&tv2,&tz2);
    printf( "%s = %g ms \n ", "Detection All time", getElapse(&tv1, &tv2) );
	unsigned int face_num = 0;
    for(vector<Bbox>::iterator it=finalBbox.begin(); it!=finalBbox.end();it++){
        if((*it).exist){
            //rectangle(cv_img, Point((*it).x1, (*it).y1), Point((*it).x2, (*it).y2), Scalar(0,0,255), 2,8,0);
            //for(int num=0;num<5;num++)circle(cv_img,Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)),3,Scalar(0,255,255), -1);
			face_num++;
        }
    }
	gettimeofday(&tv4,&tz4);
	printf( "finalBbox fize:%d\n", face_num);

	finalBbox.clear();
    imwrite("/data/local/tmp/result.jpg",cv_img);
			}
	printf( "%s = %g ms \n ", "Average Detection time", getElapse(&tv3, &tv4)/(double)iter_time );
    return 0;
}