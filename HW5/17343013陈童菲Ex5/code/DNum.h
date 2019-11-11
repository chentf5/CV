#ifndef _DNum_H
#define _DNum_H
#include <iostream>
#include <vector>
#include <map>
#include<string.h>
#include "CImg.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>  
#include <opencv2/opencv.hpp>  
 
#include <iostream> 
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <io.h> //查找文件相关函数
using namespace cv;
using namespace cimg_library;
using namespace std;
using namespace ml;

class DNum
{
public:
	DNum();
	~DNum();
	//pic1
	void drawNumImg();//划分数字块
	void detectNum(string filename);//检测上部数字
	void detectNum2(string filename);//检测标尺数字
	void detectNum3(string filename);//检测下部数字
	void trainSVM();//训练分类器
	void drawRuler();//标尺数字
	//pic2
	void drawNumImg2();//划分数字块
	void detectNum_2(string filename);//检测上部数字 可通用检测下部数字（数据）
	void detectNum2_2(string filename);//检测标尺数字
	//void detectNum3_2(string filename);//检测下部数字
	void trainSVM2();//训练分类器
	void drawRuler2();//标尺数字
	//pic3
	void drawNumImg3();//划分数字块
	void detectNum_3(string filename);//检测上部数字 可通用检测下部数字（数据）
	void detectNum2_3(string filename);//检测标尺数字
	void detectNum3_3(string filename);//检测下部数字
	void trainSVM3();//训练分类器
	void drawRuler3();//标尺数字
	void dectpoint();
private:
	Mat srcImg;
	CImg<int> srcImg2;

	Mat upImg;
	CImg<int> upImg2;
	Mat rulerImg;
	CImg<int> rulerImg2;
	Mat downImg;
	CImg<int> downImg2;

	//训练分类器
	ostringstream oss;
	int num = -1;
	Mat dealimage;
	Mat src;
	Mat yangben_gray;
	Mat yangben_thresh;
	Ptr<SVM> SVM_params;
	vector<string> numrecode;

	Ptr<SVM> SVM_params2;
	vector<string> numrecode2;
	Mat num2[1000];

	Ptr<SVM> SVM_params3;
	vector<string> numrecode3;
	Mat num3[1000];

};

#endif