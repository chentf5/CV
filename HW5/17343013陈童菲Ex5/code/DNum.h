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
#include <io.h> //�����ļ���غ���
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
	void drawNumImg();//�������ֿ�
	void detectNum(string filename);//����ϲ�����
	void detectNum2(string filename);//���������
	void detectNum3(string filename);//����²�����
	void trainSVM();//ѵ��������
	void drawRuler();//�������
	//pic2
	void drawNumImg2();//�������ֿ�
	void detectNum_2(string filename);//����ϲ����� ��ͨ�ü���²����֣����ݣ�
	void detectNum2_2(string filename);//���������
	//void detectNum3_2(string filename);//����²�����
	void trainSVM2();//ѵ��������
	void drawRuler2();//�������
	//pic3
	void drawNumImg3();//�������ֿ�
	void detectNum_3(string filename);//����ϲ����� ��ͨ�ü���²����֣����ݣ�
	void detectNum2_3(string filename);//���������
	void detectNum3_3(string filename);//����²�����
	void trainSVM3();//ѵ��������
	void drawRuler3();//�������
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

	//ѵ��������
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