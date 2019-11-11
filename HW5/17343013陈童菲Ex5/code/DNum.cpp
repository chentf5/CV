#include "DNum.h"

using namespace std;
struct con {
	double x, y;                    //轮廓位置
	int order;                      //轮廓向量contours中的第几个

	bool operator<(con &m) {
		if (x > m.x) return false;

		else return true;
	}

}con[1000];
DNum::DNum()
{

}


DNum::~DNum()
{
}

const int red[3] = { 255,0,0 };
const int green[3] = { 0.255,255 };
const int blue[3] = { 0,0,255 };
void DNum::trainSVM() {
	//核心思路：//获取一张图片后会将图片特征写入到容器中，
				//紧接着会将标签写入另一个容器中，这样就保证了特征
				//  和标签是一一对应的关系。
	////===============================读取训练数据===============================////
	const int classsum = 11;//图片共有10类，可修改
	const int imagesSum = 30;//每类有张图片，可修改	
	//训了样本图片与测试图片的尺寸应该一样
	const int imageRows = 30;//图片尺寸
	const int imageCols = 30;
	//训练数据，每一行一个训练图片
	Mat trainingData;
	//训练样本标签
	Mat labels;
	//最终的训练样本标签
	Mat clas;
	//最终的训练数据
	Mat traindata;
	//////////////////////从指定文件夹下提取图片//////////////////
	for (int p = 0; p < classsum; p++)//依次提取0到9文件夹中的图片
	{
		oss << "D:/c31/计算机视觉/HW5/hw4/hw4/numberImage/";
		num += 1;//num从0到9
		
		int label = num;
		oss << num << "/*.bmp";//图片名字后缀，oss可以结合数字与字符串
		string pattern = oss.str();//oss.str()输出oss字符串，并且赋给pattern
		oss.str("");//每次循环后把oss字符串清空
		vector<Mat> input_images;
		vector<String> input_images_name;
		glob(pattern, input_images_name, false);
		//为false时，仅仅遍历指定文件夹内符合模式的文件，当为true时，会同时遍历指定文件夹的子文件夹
		//此时input_images_name存放符合条件的图片地址
		int all_num = input_images_name.size();
		//文件下总共有几个图片
		//cout << num << ":总共有" << all_num << "个图片待测试" << endl;

		for (int i = 0; i < imagesSum; i++)//依次循环遍历每个文件夹中的图片
		{
			cvtColor(imread(input_images_name[i]), yangben_gray, COLOR_BGR2GRAY);//灰度变换
			threshold(yangben_gray, yangben_thresh, 0, 255, THRESH_OTSU);//二值化
			//循环读取每张图片并且依次放在vector<Mat> input_images内
			input_images.push_back(yangben_thresh);
			dealimage = input_images[i];


			//注意：我们简单粗暴将整个图的所有像素作为了特征，因为我们关注更多的是整个的训练过程
			//，所以选择了最简单的方式完成特征提取工作，除此中外，
			//特征提取的方式有很多，比如LBP，HOG等等
			//我们利用reshape()函数完成特征提取,
			//eshape(1, 1)的结果就是原图像对应的矩阵将被拉伸成一个一行的向量，作为特征向量。 
			dealimage = dealimage.reshape(1, 1);//图片序列化
			trainingData.push_back(dealimage);//序列化后的图片依次存入
			labels.push_back(label);//把每个图片对应的标签依次存入
		}
	}
	//图片数据和标签转变下
	Mat(trainingData).copyTo(traindata);//复制
	traindata.convertTo(traindata, CV_32FC1);//更改图片数据的类型，必要，不然会出错
	Mat(labels).copyTo(clas);//复制


	////===============================创建SVM模型===============================////
	// 创建分类器并设置参数
	SVM_params = SVM::create();
	SVM_params->setType(SVM::C_SVC);//C_SVC用于分类，C_SVR用于回归
	SVM_params->setKernel(SVM::LINEAR);  //LINEAR线性核函数。SIGMOID为高斯核函数

	SVM_params->setDegree(0);//核函数中的参数degree,针对多项式核函数;
	SVM_params->setGamma(1);//核函数中的参数gamma,针对多项式/RBF/SIGMOID核函数; 
	SVM_params->setCoef0(0);//核函数中的参数,针对多项式/SIGMOID核函数；
	SVM_params->setC(1);//SVM最优问题参数，设置C-SVC，EPS_SVR和NU_SVR的参数；
	SVM_params->setNu(0);//SVM最优问题参数，设置NU_SVC， ONE_CLASS 和NU_SVR的参数； 
	SVM_params->setP(0);//SVM最优问题参数，设置EPS_SVR 中损失函数p的值. 
	//结束条件，即训练1000次或者误差小于0.01结束
	SVM_params->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01));

	//训练数据和标签的结合
	Ptr<TrainData> tData = TrainData::create(traindata, ROW_SAMPLE, clas);

	// 训练分类器
	SVM_params->train(tData);//训练

	//保存模型
	//SVM_params->save("C:/Users/zhang/Desktop/opencv——实例/小案例/车牌检测/基于adaboost机器学习/字符识别svm.xml");
	cout << "训练好了！！！" << endl;

}

CImg<int> draw_rect(CImg<double> img,int x0, int y0, int x1, int y1,int color) {
	if (color == 1) {
		img.draw_rectangle(x0, y0, x0, y1, red, 1);
		img.draw_rectangle(x0, y1, x1, y1, red, 1);
		img.draw_rectangle(x1, y0, x1, y1, red, 1);
		img.draw_rectangle(x0, y0, x1, y0, red, 1);
	}
	else if (color == 2) {
		img.draw_rectangle(x0, y0, x0, y1, green, 1);
		img.draw_rectangle(x0, y1, x1, y1, green, 1);
		img.draw_rectangle(x1, y0, x1, y1, green, 1);
		img.draw_rectangle(x0, y0, x1, y0, green, 1);
	}
	else if (color == 3) {
		img.draw_rectangle(x0, y0, x0, y1, blue, 1);
		img.draw_rectangle(x0, y1, x1, y1, blue, 1);
		img.draw_rectangle(x1, y0, x1, y1, blue, 1);
		img.draw_rectangle(x0, y0, x1, y0, blue, 1);
	}
	return img;
	
}
void getnewImg(CImg<double> img, int x0, int y0, int x1, int y1,const char *filename) {
	//img1.load_bmp("result1.bmp");

	CImg<double> newImg;
	int wid = x1 - x0;
	int hei = y1 - y0;
	newImg.resize(wid, hei, 1, 1, 0);

	for (int i = x0; i < x1; i++) {
		for (int j = y0; j < y1; j++) {
			newImg(i - x0, j - y0) = img(i, j);
		}
	}
	newImg.display("newImg");
	newImg.save(filename);

}


struct result {
	double bi;
	int num;

	bool operator<(result &m) {
		if (bi < m.bi)return true;
		else return false;
	}
}result[10];


Mat num[1000];
Mat sample;
void deal(Mat &src, int order);
double compare(Mat &src, Mat &sample);
void Threshold(Mat &src, Mat &sample, int m);

void DNum::drawNumImg() {
	
	srcImg2.load_bmp("1.bmp");
	srcImg2.display("0");
	CImg<int> temp;
	temp.resize(srcImg2.width(), srcImg2.height(), 1, 1, 0);
	temp = srcImg2;
	int x0[4] = {139,200,180,1430};
	int y0[4] = {120,1400,1460,1460};
	int x1[4] = {2080,1936,232,2000};
	int y1[4] = {183,1427,1480,1500};
	int color[4] = { 1,3,2,2 };
	for (int i = 0; i < 4; i++) {
		temp = draw_rect(temp, x0[i], y0[i], x1[i], y1[i], color[i]);
		
	}
	getnewImg(srcImg2, x0[0], y0[0], x1[0], y1[0],"num1.bmp");
	getnewImg(srcImg2, x0[1], y0[1], x1[1], y1[1], "num2.bmp");
	getnewImg(srcImg2, x0[2], y0[2], x1[2], y1[2], "num3.bmp");
	getnewImg(srcImg2, x0[3], y0[3], x1[3], y1[3], "num4.bmp");
	temp.display("drawNumImg");
	//srcImg2.save("drawNumImg.bmp");
	
}
Mat db[11];
const int dbLen = 11;

void DNum::detectNum(string filename) {
	/*Mat srcImage = imread(filename);
	Mat grayImage = Mat::zeros(srcImage.size(), CV_8UC1);
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	Mat dstImage = Mat::zeros(grayImage.size(), grayImage.type());
	adaptiveThreshold(grayImage, dstImage, 255, ADAPTIVE_THRESH_MEAN_C, 0, 7, 0);
	namedWindow("img", 1);
	imshow("img", dstImage);*/
	Mat srcImage = imread(filename);
	Mat dstImage, grayImage, Image;
	srcImage.copyTo(dstImage);
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	threshold(grayImage, Image, 48, 255, THRESH_BINARY_INV);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(Image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	int i = 0;

	vector<vector<Point>>::iterator It;
	vector<vector<Point>>::iterator temp;
	vector<vector<Point>>::iterator itor2;
	Mat copyImage = Image.clone();
	Rect rect[1000];
	for (It = contours.begin(); It < contours.end(); It++) {

		//画出可包围数字的最小矩形
		int num = (*It).size();
		if (num > 100) {
			continue;
		}
		Point2f vertex[4];
		rect[i] = boundingRect(*It);
		vertex[0] = rect[i].tl();                                                           //矩阵左上角的点
		vertex[1].x = (float)rect[i].tl().x, vertex[1].y = (float)rect[i].br().y;           //矩阵左下方的点
		vertex[2] = rect[i].br();                                                           //矩阵右下角的点
		vertex[3].x = (float)rect[i].br().x, vertex[3].y = (float)rect[i].tl().y;           //矩阵右上方的点

		for (int j = 0; j < 4; j++)
			line(dstImage, vertex[j], vertex[(j + 1) % 4], Scalar(0, 0, 255), 1);

		con[i].x = vertex[0].x;                  //根据中心点判断图像的位置
		con[i].y = (vertex[1].y + vertex[2].y) / 2.0;
		con[i].order = i;
		i++;
	}
	//namedWindow("number", WINDOW_AUTOSIZE);
	//imshow("number", dstImage);
	//sort(con, con + i);
	/*
	for (int t = i - 1, name = 0; t >= 0; t--, name++)
	{
		int k = con[t].order;
		Mat tempImg;
		resize(copyImage(rect[k]), tempImg, Size(30, 30));
		ostringstream fileName;
		fileName << "numberImage/" << name << ".bmp";
		imwrite(fileName.str(), tempImg);
	}*/
	//imwrite("number2.bmp", dstImage);
	//载入数据库 10张数字图片作为模板

	for (int i = 0; i < 10; i++)
	{
		ostringstream fileName;
		fileName << "number/" << i << ".bmp";
		db[i] = imread(fileName.str());
	}
	ostringstream fileName1;
	fileName1 << "number/p.bmp";
	db[10] = imread(fileName1.str());
	
	
	/*for (int j = 0; j < i; j++) {
		int k = con[j].order;
		//cout << con[j].order << " " << con[j].x << " " << con[j].y;
		int rows = copyImage(rect[j]).rows;
		int cols = copyImage(rect[j]).cols;
		int same = 0;
		
		if (cols * rows <=22) {
			con[j].order = -1;
			cout << "delete";
		}
		cout <<  endl;
	}
	*/
	//真正检测函数
	int four = 0;
	for (int j = i-1; j >=0 ; j--)
	{

		//调整矩阵 与数据库图片大小一致
		int k = con[j].order;
		//cout << con[j].order << " " << con[j].x << " " << con[j].y << endl;
		int rows1 = copyImage(rect[k]).rows;
		int cols1 = copyImage(rect[k]).cols;
		if (rows1*cols1 <= 20) {
			//cout << ".";
			continue;
		}
		Mat tempRect = Mat::zeros(db[0].size(), db[0].type());
		resize(copyImage(rect[k]), tempRect, tempRect.size());
		int rows = tempRect.rows;
		int cols = tempRect.cols*tempRect.channels();
		//cvtColor(tempRect, tempRect, COLOR_BGR2GRAY);
		threshold(tempRect, tempRect, 0, 255, THRESH_OTSU);
		tempRect = tempRect.reshape(1, 1);
		Mat inputtemp;
		inputtemp.push_back(tempRect);
		inputtemp.convertTo(inputtemp, CV_32FC1);//更改图片数据的类型，必要，不然会出错

		float r = SVM_params->predict(inputtemp);   //对所有行进行预测
		cout << r;


		/*
		//每个rect在数据库中进行匹配,找出最相近的值
		int Matchresult = 0;//最的小匹配值
		int MatchIndex = 0;//最小匹配值对应的的db序号

		for (int u = 0; u < dbLen; u++)
		{
			int same = 0;

			for (int v = 0; v < rows; v++)
			{
				uchar*data1 = tempRect.ptr<uchar>(v);
				uchar*data2 = db[u].ptr<uchar>(v);

				for (int w = 0; w < cols; w++)
				{
					if (data1[w] == data2[w])
						same++;
				}
			}

			if (same > Matchresult)
			{
				Matchresult = same;
				MatchIndex = u;
			}

			//如果这张图片完全一致 则不匹配其他的图片
			
		}*/
		//输出匹配的结果
		//cout << j << ":";
		
		//cout << MatchIndex;
		//deal(tempRect, j + 1);

		//小数点逻辑关系
		//num1
		
		four++;
		if (four >= 4) {
				four = 0;
				cout << endl;
		}
		else if (four == 1) {
			cout << ".";
		}
		
	}
	
}

void DNum::detectNum2(string filename) {
	/*Mat srcImage = imread(filename);
	Mat grayImage = Mat::zeros(srcImage.size(), CV_8UC1);
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	Mat dstImage = Mat::zeros(grayImage.size(), grayImage.type());
	adaptiveThreshold(grayImage, dstImage, 255, ADAPTIVE_THRESH_MEAN_C, 0, 7, 0);
	namedWindow("img", 1);
	imshow("img", dstImage);*/
	Mat srcImage = imread(filename);
	Mat dstImage, grayImage, Image;
	srcImage.copyTo(dstImage);
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	threshold(grayImage, Image, 48, 255, THRESH_BINARY_INV);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(Image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	int i = 0;

	vector<vector<Point>>::iterator It;
	vector<vector<Point>>::iterator temp;
	vector<vector<Point>>::iterator itor2;
	Mat copyImage = Image.clone();
	Rect rect[1000];
	for (It = contours.begin(); It < contours.end(); It++) {

		//画出可包围数字的最小矩形
		int num = (*It).size();
		if (num > 100) {
			continue;
		}
		Point2f vertex[4];
		rect[i] = boundingRect(*It);
		vertex[0] = rect[i].tl();                                                           //矩阵左上角的点
		vertex[1].x = (float)rect[i].tl().x, vertex[1].y = (float)rect[i].br().y;           //矩阵左下方的点
		vertex[2] = rect[i].br();                                                           //矩阵右下角的点
		vertex[3].x = (float)rect[i].br().x, vertex[3].y = (float)rect[i].tl().y;           //矩阵右上方的点

		for (int j = 0; j < 4; j++)
			line(dstImage, vertex[j], vertex[(j + 1) % 4], Scalar(0, 0, 255), 1);

		con[i].x = vertex[0].x;                  //根据中心点判断图像的位置
		con[i].y = (vertex[1].y + vertex[2].y) / 2.0;
		con[i].order = i;
		i++;
	}
	//namedWindow("number", WINDOW_AUTOSIZE);
	//imshow("number", dstImage);
	sort(con, con + i);
	/*
	for (int t = i - 1, name = 0; t >= 0; t--, name++)
	{
		int k = con[t].order;
		Mat tempImg;
		resize(copyImage(rect[k]), tempImg, Size(30, 30));
		ostringstream fileName;
		fileName << "numberImage/" << name << ".bmp";
		imwrite(fileName.str(), tempImg);
	}*/
	//imwrite("number2.bmp", dstImage);
	//载入数据库 10张数字图片作为模板

	for (int i = 0; i < 10; i++)
	{
		ostringstream fileName;
		fileName << "number/" << i << ".bmp";
		db[i] = imread(fileName.str());
	}
	ostringstream fileName1;
	fileName1 << "number/p.bmp";
	db[10] = imread(fileName1.str());

	/*
	for (int j = 0; j < i; j++) {
		int k = con[j].order;
		//cout << con[j].order << " " << con[j].x << " " << con[j].y;
		int rows = copyImage(rect[j]).rows;
		int cols = copyImage(rect[j]).cols;
		int same = 0;

		if (cols * rows <= 22) {
			con[j].order = -1;
			cout << "delete";
		}
		cout << endl;
	}
	*/
	int four = 0;
	for (int j = 0; j < i; j++)
	{

		//调整矩阵 与数据库图片大小一致
		int k = con[j].order;
		//cout << con[j].order << " " << con[j].x << " " << con[j].y << endl;
		int rows1 = copyImage(rect[k]).rows;
		int cols1 = copyImage(rect[k]).cols;
		if (rows1*cols1 <= 24) {
			cout << ".";
			continue;
		}
		Mat tempRect = Mat::zeros(db[0].size(), db[0].type());
		resize(copyImage(rect[k]), tempRect, tempRect.size());
		int rows = tempRect.rows;
		int cols = tempRect.cols*tempRect.channels();
		//cvtColor(tempRect, tempRect, COLOR_BGR2GRAY);
		threshold(tempRect, tempRect, 0, 255, THRESH_OTSU);
		tempRect = tempRect.reshape(1, 1);
		Mat inputtemp;
		inputtemp.push_back(tempRect);
		inputtemp.convertTo(inputtemp, CV_32FC1);//更改图片数据的类型，必要，不然会出错

		float r = SVM_params->predict(inputtemp);   //对所有行进行预测
		cout << r;




		//小数点逻辑关系
		//num1
		//num2
		four++;
		if (four >= 2) {
			four = 0;
			cout << endl;
		}
		
	}

}
void DNum::detectNum3(string filename) {
	/*Mat srcImage = imread(filename);
	Mat grayImage = Mat::zeros(srcImage.size(), CV_8UC1);
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	Mat dstImage = Mat::zeros(grayImage.size(), grayImage.type());
	adaptiveThreshold(grayImage, dstImage, 255, ADAPTIVE_THRESH_MEAN_C, 0, 7, 0);
	namedWindow("img", 1);
	imshow("img", dstImage);*/
	Mat srcImage = imread(filename);
	Mat dstImage, grayImage, Image;
	srcImage.copyTo(dstImage);
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	threshold(grayImage, Image, 48, 255, THRESH_BINARY_INV);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(Image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	int i = 0;

	vector<vector<Point>>::iterator It;
	vector<vector<Point>>::iterator temp;
	vector<vector<Point>>::iterator itor2;
	Mat copyImage = Image.clone();
	Rect rect[1000];
	for (It = contours.begin(); It < contours.end(); It++) {

		//画出可包围数字的最小矩形
		int num = (*It).size();
		if (num > 100) {
			continue;
		}
		Point2f vertex[4];
		rect[i] = boundingRect(*It);
		vertex[0] = rect[i].tl();                                                           //矩阵左上角的点
		vertex[1].x = (float)rect[i].tl().x, vertex[1].y = (float)rect[i].br().y;           //矩阵左下方的点
		vertex[2] = rect[i].br();                                                           //矩阵右下角的点
		vertex[3].x = (float)rect[i].br().x, vertex[3].y = (float)rect[i].tl().y;           //矩阵右上方的点

		for (int j = 0; j < 4; j++)
			line(dstImage, vertex[j], vertex[(j + 1) % 4], Scalar(0, 0, 255), 1);

		con[i].x = vertex[0].x;                  //根据中心点判断图像的位置
		con[i].y = (vertex[1].y + vertex[2].y) / 2.0;
		con[i].order = i;
		i++;
	}
	//namedWindow("number", WINDOW_AUTOSIZE);
	//imshow("number", dstImage);
	sort(con, con + i);
	/*
	for (int t = i - 1, name = 0; t >= 0; t--, name++)
	{
		int k = con[t].order;
		Mat tempImg;
		resize(copyImage(rect[k]), tempImg, Size(30, 30));
		ostringstream fileName;
		fileName << "numberImage/" << name << ".bmp";
		imwrite(fileName.str(), tempImg);
	}*/
	//imwrite("number4.bmp", dstImage);
	//载入数据库 10张数字图片作为模板

	for (int i = 0; i < 10; i++)
	{
		ostringstream fileName;
		fileName << "number/" << i << ".bmp";
		db[i] = imread(fileName.str());
	}
	ostringstream fileName1;
	fileName1 << "number/p.bmp";
	db[10] = imread(fileName1.str());

	/*
	for (int j = 0; j < i; j++) {
		int k = con[j].order;
		//cout << con[j].order << " " << con[j].x << " " << con[j].y;
		int rows = copyImage(rect[j]).rows;
		int cols = copyImage(rect[j]).cols;
		int same = 0;

		if (cols * rows <= 22) {
			con[j].order = -1;
			cout << "delete";
		}
		cout << endl;
	}
	*/
	int four = 0;
	string tempNum = "";
	for (int j = 0; j < i; j++)
	{
		
		//调整矩阵 与数据库图片大小一致
		int k = con[j].order;
		//cout << con[j].order << " " << con[j].x << " " << con[j].y << endl;
		int rows1 = copyImage(rect[k]).rows;
		int cols1 = copyImage(rect[k]).cols;
		
		Mat tempRect = Mat::zeros(db[0].size(), db[0].type());
		resize(copyImage(rect[k]), tempRect, tempRect.size());
		int rows = tempRect.rows;
		int cols = tempRect.cols*tempRect.channels();
		//cvtColor(tempRect, tempRect, COLOR_BGR2GRAY);
		threshold(tempRect, tempRect, 0, 255, THRESH_OTSU);
		tempRect = tempRect.reshape(1, 1);
		Mat inputtemp;
		inputtemp.push_back(tempRect);
		inputtemp.convertTo(inputtemp, CV_32FC1);//更改图片数据的类型，必要，不然会出错
		
		float r = SVM_params->predict(inputtemp);   //对所有行进行预测
		if (r == 10) {
			cout << ".";
			tempNum += ".";
		}
		else { 
			cout << r; 
			tempNum += int(r)+'0';
		}




		//小数点逻辑关系
		//num1
		//num2
		four++;
		if (four >= 4) {
			four = 0;
			cout << endl;
			numrecode.push_back(tempNum);
			tempNum = "";
		}

	}

}
void Threshold(Mat &src, Mat &sample, int m)
{
	cvtColor(sample, sample, COLOR_BGR2GRAY);
	threshold(sample, sample, 48, 255, THRESH_BINARY_INV);
	result[m].bi = compare(src, sample);
	result[m].num = m;
}

void deal(Mat &src, int order)
{

	sample = imread("number/0.bmp");
	Threshold(src, sample, 0);

	sample = imread("number/1.bmp");
	Threshold(src, sample, 1);

	sample = imread("number/2.bmp");
	Threshold(src, sample, 2);

	sample = imread("number/3.bmp");
	Threshold(src, sample, 3);

	sample = imread("number/4.bmp");
	Threshold(src, sample, 4);

	sample = imread("number/5.bmp");
	Threshold(src, sample, 5);

	sample = imread("number/6.bmp");
	Threshold(src, sample, 6);

	sample = imread("number/7.bmp");
	Threshold(src, sample, 7);

	sample = imread("number/8.bmp");
	Threshold(src, sample, 8);

	sample = imread("number/9.bmp");
	Threshold(src, sample, 9);

	sort(result, result + 10);

	if (result[9].bi > 0.6) {
		//cout << "第" << order << "个数字为 " << result[9].num << endl;
		cout << result[9].num;
		//cout << "识别精度为 " << result[9].bi << endl;
	}
	//else cout << "第" << order << "个数字无法识别" << endl;
}

double compare(Mat &src, Mat &sample)
{
	double same = 0.0, difPoint = 0.0;
	//Mat now;
	//resize(sample, now, src.size());
	int row = sample.rows;
	int col = sample.cols *  sample.channels();
	for (int i = 0; i < 1; i++) {
		uchar * data1 = src.ptr<uchar>(i);
		uchar * data2 = sample.ptr<uchar>(i);
		for (int j = 0; j < row * col; j++) {
			int  a = data1[j];
			int b = data2[j];
			if (a == b)same++;
			else difPoint++;
		}
	}
	return same / (same + difPoint);
}



void DNum::drawRuler() {
	CImg<int> temp;
	temp.load_bmp("2.bmp");
	cimg_forXY(temp, x, y) {
		if (temp(x, y) == 0) {
			temp(x, y,0) = 255;
			temp(x, y, 1) = 255;
			temp(x, y, 2) = 255;
		}
		else {
			temp(x, y, 0) = 0;
			temp(x, y, 1) = 0;
			temp(x, y, 2) = 0;
		}
	}
	temp.display("1");
	temp.dilate(3);
	temp.display("1");
	//getnewImg(temp, x0, y0, x1, y1, "2.bmp");

	//找到标尺固定点
	int maxp = 0;
	int rulerx = 0;
	int rulery = 0;
	
	cimg_forY(temp, y) {
		int point = 0;
		int tx = 0;
		cimg_forX(temp, x) {
			if (temp(x, y) == 255) {
				point++;
				tx = x;
			}
		}
		if (maxp < point) {
			maxp = point;
			rulery = y;
			rulerx = tx;
		}
	}
	cout << rulerx << " " << rulery << endl;
	int flag = 0;
	int yunx[2];
	for (int i = rulerx; i >= 0; i--) {
		int point = 0;
		for (int j = rulery; j < rulery + 10; j++) {
			if (temp(i, j) == 255) {
				point++;
				
			}
		}
		if (point > 3) {
			yunx[flag] = i;
			flag++;
			i -= 3;
			if (flag >= 2) break;
		}
	}
	cout << yunx[0]  << " "<< yunx[1] << endl;
	
	int tempy = 91;
	vector<int> arrayi;
	for (int i = 0; i < temp.width(); i++) {
		if (temp(i, tempy) == 255) {
			while (true) {
				int point = 0;
				for (int j = tempy; j > tempy - 5; j--) {
					if (temp(i, j) == 255) {
						point++;
					}
				}
				if (point > 3) {
					arrayi.push_back(i);
					i += 5;
				}
				i++;
				if (temp(i, tempy) == 0) {
					arrayi.push_back(-1);
					break;
				}
			}
		}
		
		
	}
	int sizt = arrayi.size();
	for (int i = 0; i < sizt; i++) {
		cout << arrayi[i] << " ";
	}
	cout << endl;




	temp.display("1");
	int bili = yunx[0] - yunx[1];
	int sizt_m = arrayi.size();
	vector<double> juli;
	for (int i = 0; i < sizt_m; i++) {
		if (arrayi[i] != -1) {
			double len = yunx[0] - arrayi[i];
			len = (len / bili)*0.1+0.6;
			juli.push_back(len);
		}
		else {
			juli.push_back(-1);
		}
		
	}
	int s = 0,e = 0;
	//vector<double> num; 



	double numk[8] = {
		1.00, 1.06, 5.20,5.37,7.31,4.28,1.07,6.28
	};
	int k = 0;
	for (int i = 0; i < juli.size(); i++) {
		if (juli[i] == -1) {
			e = i;
			for (int j = s; j < e-1; j++) {
				cout << numrecode[k] << " ";
				cout << setprecision(4) << juli[j] << " "<< setprecision(4) << juli[j + 1]<<  endl;
				k++;
			}
			s = i + 1;
		}
	}

}


void DNum::drawNumImg2() {
	CImg<int> srcImg2;
	srcImg2.load_bmp("H-Image1.bmp");
	//temp.display();
	CImg<int> temp;
	temp.resize(srcImg2.width(), srcImg2.height(), 1, 1, 0);
	temp = srcImg2;
	int x0[4] = { 125,40,40,40 };
	int y0[4] = { 50,1235,1320,1220 };
	int x1[4] = { 2220,2250,2250,2250 };
	int y1[4] = { 115,1300,1350,1350};
	int color[4] = { 1,3,2,2 };
	for (int i = 0; i < 3; i++) {
		temp = draw_rect(temp, x0[i], y0[i], x1[i], y1[i], color[i]);

	}
	//getnewImg(srcImg2, x0[0], y0[0], x1[0], y1[0], "num1_2.bmp");
	//getnewImg(srcImg2, x0[1], y0[1], x1[1], y1[1], "num2_2.bmp");
	//getnewImg(srcImg2, x0[2], y0[2], x1[2], y1[2], "num3_2.bmp");
	//getnewImg(srcImg2, x0[3], y0[3], x1[3], y1[3], "ruler_2.bmp");
	temp.display("drawNumImg");
	temp.save("r2_2.bmp");
}

void DNum::detectNum_2(string filename) {
	Mat srcImage = imread(filename);
	Mat dstImage, grayImage, Image;
	srcImage.copyTo(dstImage);
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	threshold(grayImage, Image, 48, 255, THRESH_BINARY_INV);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(Image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	int i = 0;

	vector<vector<Point>>::iterator It;
	vector<vector<Point>>::iterator temp;
	vector<vector<Point>>::iterator itor2;
	Mat copyImage = Image.clone();
	Rect rect[1000];
	for (It = contours.begin(); It < contours.end(); It++) {

		//画出可包围数字的最小矩形
		int num = (*It).size();
		if (num > 100) {
			continue;
		}
		Point2f vertex[4];
		rect[i] = boundingRect(*It);
		vertex[0] = rect[i].tl();                                                           //矩阵左上角的点
		vertex[1].x = (float)rect[i].tl().x, vertex[1].y = (float)rect[i].br().y;           //矩阵左下方的点
		vertex[2] = rect[i].br();                                                           //矩阵右下角的点
		vertex[3].x = (float)rect[i].br().x, vertex[3].y = (float)rect[i].tl().y;           //矩阵右上方的点

		for (int j = 0; j < 4; j++)
			line(dstImage, vertex[j], vertex[(j + 1) % 4], Scalar(0, 0, 255), 1);

		con[i].x = vertex[0].x;                  //根据中心点判断图像的位置
		con[i].y = vertex[1].y;
		con[i].order = i;
		i++;
	}
	//imwrite("number2.bmp", dstImage);
	//namedWindow("number", WINDOW_AUTOSIZE);
	//imshow("number", dstImage);
	//sort(con, con + i);
	/*
	for (int t = i - 1, name = 0; t >= 0; t--, name++)
	{
		int k = con[t].order;
		Mat tempImg;
		resize(copyImage(rect[k]), tempImg, Size(30, 30));
		ostringstream fileName;
		fileName << "numberImage/" << name << ".bmp";
		imwrite(fileName.str(), tempImg);
	}
	

	*/
	/*for (int j = 0; j < i; j++) {
		int k = con[j].order;
		//cout << con[j].order << " " << con[j].x << " " << con[j].y;
		int rows = copyImage(rect[j]).rows;
		int cols = copyImage(rect[j]).cols;
		int same = 0;

		if (cols * rows <=22) {
			con[j].order = -1;
			cout << "delete";
		}
		cout <<  endl;
	}
	*/
	//真正检测函数
	string tempNum = "";
	int four = 0;
	for (int j = i - 1; j >= 0; j--)
	{

		//调整矩阵 与数据库图片大小一致
		int k = con[j].order;
		//cout << con[j].order << " " << con[j].x << " " << con[j].y << endl;
		int rows1 = copyImage(rect[k]).rows;
		int cols1 = copyImage(rect[k]).cols;
		if (rows1*cols1 <= 20) {
			//cout << ".";
			continue;
		}
		Mat tempRect = Mat::zeros(Size(30, 30), CV_32FC1);
		resize(copyImage(rect[k]), tempRect, tempRect.size());
		int rows = tempRect.rows;
		int cols = tempRect.cols*tempRect.channels();
		//cvtColor(tempRect, tempRect, COLOR_BGR2GRAY);
		threshold(tempRect, tempRect, 0, 255, THRESH_OTSU);
		tempRect = tempRect.reshape(1, 1);
		Mat inputtemp;
		inputtemp.push_back(tempRect);
		inputtemp.convertTo(inputtemp, CV_32FC1);//更改图片数据的类型，必要，不然会出错

		float r = SVM_params2->predict(inputtemp);   //对所有行进行预测
		cout << r;
		tempNum += int(r) + '0';

		//小数点逻辑关系
		//num1

		four++;
		if (four >= 3) {
			four = 0;
			cout << endl;
			numrecode2.push_back(tempNum);
			tempNum = "";
		}
		else if (four == 1) {
			cout << ".";
			tempNum += ".";
		}

	}
	
}

void DNum::trainSVM2() {
	//核心思路：//获取一张图片后会将图片特征写入到容器中，
			//紧接着会将标签写入另一个容器中，这样就保证了特征
			//  和标签是一一对应的关系。
////===============================读取训练数据===============================////
	const int classsum = 11;//图片共有10类，可修改
	const int imagesSum = 15;//每类有张图片，可修改	
	//训了样本图片与测试图片的尺寸应该一样
	const int imageRows = 30;//图片尺寸
	const int imageCols = 30;
	//训练数据，每一行一个训练图片
	Mat trainingData;
	//训练样本标签
	Mat labels;
	//最终的训练样本标签
	Mat clas;
	//最终的训练数据
	Mat traindata;
	//////////////////////从指定文件夹下提取图片//////////////////
	for (int p = 0; p < classsum; p++)//依次提取0到9文件夹中的图片
	{
		oss << "D:/c31/计算机视觉/HW5/hw4/hw4/numberImage/2";
		num += 1;//num从0到9

		int label = num;
		oss << num << "/*.bmp";//图片名字后缀，oss可以结合数字与字符串
		string pattern = oss.str();//oss.str()输出oss字符串，并且赋给pattern
		oss.str("");//每次循环后把oss字符串清空
		vector<Mat> input_images;
		vector<String> input_images_name;
		glob(pattern, input_images_name, false);
		//为false时，仅仅遍历指定文件夹内符合模式的文件，当为true时，会同时遍历指定文件夹的子文件夹
		//此时input_images_name存放符合条件的图片地址
		int all_num = input_images_name.size();
		//文件下总共有几个图片
		//cout << num << ":总共有" << all_num << "个图片待测试" << endl;

		for (int i = 0; i < imagesSum; i++)//依次循环遍历每个文件夹中的图片
		{
			cvtColor(imread(input_images_name[i]), yangben_gray, COLOR_BGR2GRAY);//灰度变换
			threshold(yangben_gray, yangben_thresh, 0, 255, THRESH_OTSU);//二值化
			//循环读取每张图片并且依次放在vector<Mat> input_images内
			input_images.push_back(yangben_thresh);
			dealimage = input_images[i];


			//注意：我们简单粗暴将整个图的所有像素作为了特征，因为我们关注更多的是整个的训练过程
			//，所以选择了最简单的方式完成特征提取工作，除此中外，
			//特征提取的方式有很多，比如LBP，HOG等等
			//我们利用reshape()函数完成特征提取,
			//eshape(1, 1)的结果就是原图像对应的矩阵将被拉伸成一个一行的向量，作为特征向量。 
			dealimage = dealimage.reshape(1, 1);//图片序列化
			trainingData.push_back(dealimage);//序列化后的图片依次存入
			labels.push_back(label);//把每个图片对应的标签依次存入
		}
	}
	//图片数据和标签转变下
	Mat(trainingData).copyTo(traindata);//复制
	traindata.convertTo(traindata, CV_32FC1);//更改图片数据的类型，必要，不然会出错
	Mat(labels).copyTo(clas);//复制


	////===============================创建SVM模型===============================////
	// 创建分类器并设置参数
	SVM_params2 = SVM::create();
	SVM_params2->setType(SVM::C_SVC);//C_SVC用于分类，C_SVR用于回归
	SVM_params2->setKernel(SVM::LINEAR);  //LINEAR线性核函数。SIGMOID为高斯核函数

	SVM_params2->setDegree(0);//核函数中的参数degree,针对多项式核函数;
	SVM_params2->setGamma(1);//核函数中的参数gamma,针对多项式/RBF/SIGMOID核函数; 
	SVM_params2->setCoef0(0);//核函数中的参数,针对多项式/SIGMOID核函数；
	SVM_params2->setC(1);//SVM最优问题参数，设置C-SVC，EPS_SVR和NU_SVR的参数；
	SVM_params2->setNu(0);//SVM最优问题参数，设置NU_SVC， ONE_CLASS 和NU_SVR的参数； 
	SVM_params2->setP(0);//SVM最优问题参数，设置EPS_SVR 中损失函数p的值. 
	//结束条件，即训练1000次或者误差小于0.01结束
	SVM_params2->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01));

	//训练数据和标签的结合
	Ptr<TrainData> tData = TrainData::create(traindata, ROW_SAMPLE, clas);

	// 训练分类器
	SVM_params2->train(tData);//训练

	//保存模型
	//SVM_params->save("C:/Users/zhang/Desktop/opencv——实例/小案例/车牌检测/基于adaboost机器学习/字符识别svm.xml");
	cout << "训练好了！！！" << endl;
}

void DNum::detectNum2_2(string filename)	{
	Mat srcImage = imread(filename);
	Mat dstImage, grayImage, Image;
	srcImage.copyTo(dstImage);
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	threshold(grayImage, Image, 48, 255, THRESH_BINARY_INV);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(Image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	int i = 0;

	vector<vector<Point>>::iterator It;
	vector<vector<Point>>::iterator temp;
	vector<vector<Point>>::iterator itor2;
	Mat copyImage = Image.clone();
	Rect rect[1000];
	for (It = contours.begin(); It < contours.end(); It++) {

		//画出可包围数字的最小矩形
		int num = (*It).size();
		if (num > 100) {
			continue;
		}
		Point2f vertex[4];
		rect[i] = boundingRect(*It);
		vertex[0] = rect[i].tl();                                                           //矩阵左上角的点
		vertex[1].x = (float)rect[i].tl().x, vertex[1].y = (float)rect[i].br().y;           //矩阵左下方的点
		vertex[2] = rect[i].br();                                                           //矩阵右下角的点
		vertex[3].x = (float)rect[i].br().x, vertex[3].y = (float)rect[i].tl().y;           //矩阵右上方的点

		for (int j = 0; j < 4; j++)
			line(dstImage, vertex[j], vertex[(j + 1) % 4], Scalar(0, 0, 255), 1);

		con[i].x = vertex[0].x;                  //根据中心点判断图像的位置
		con[i].y = vertex[1].y;
		con[i].order = i;
		i++;
	}
	imwrite("number3.bmp", dstImage);
	sort(con, con + i);
	//真正检测函数

	int four = 0;
	for (int j = 0; j < i; j++)
	{

		//调整矩阵 与数据库图片大小一致
		int k = con[j].order;
		//cout << con[j].order << " " << con[j].x << " " << con[j].y << endl;
		int rows1 = copyImage(rect[k]).rows;
		int cols1 = copyImage(rect[k]).cols;
		Mat tempRect = Mat::zeros(Size(30, 30), CV_32FC1);
		resize(copyImage(rect[k]), tempRect, tempRect.size());
		int rows = tempRect.rows;
		int cols = tempRect.cols*tempRect.channels();
		//cvtColor(tempRect, tempRect, COLOR_BGR2GRAY);
		threshold(tempRect, tempRect, 0, 255, THRESH_OTSU);
		tempRect = tempRect.reshape(1, 1);
		Mat inputtemp;
		inputtemp.push_back(tempRect);
		inputtemp.convertTo(inputtemp, CV_32FC1);//更改图片数据的类型，必要，不然会出错

		float r = SVM_params2->predict(inputtemp);   //对所有行进行预测
		if (r == 10) {
			cout << ".";
			//tempNum += ".";
		}
		else {
			cout << r;
			//tempNum += int(r) + '0';
		}





		four++;
		if (four >= 3) {
			four = 0;
			cout << endl;
		}
	}
}

void DNum::drawRuler2() {
	CImg<int> temp;
	temp.load_bmp("ruler_2.bmp");
	cimg_forXY(temp, x, y) {
		if (temp(x, y) == 255) {
			temp(x, y, 0) = 0;
			temp(x, y, 1) = 0;
			temp(x, y, 2) = 0;
		}
		else {
			temp(x, y, 0) = 255;
			temp(x, y, 1) = 255;
			temp(x, y, 2) = 255;
		}
	}
	//temp.display("1");
	//temp.dilate(3);
	//temp.display("1");
	//getnewImg(temp, x0, y0, x1, y1, "2.bmp");

	//找到标尺固定点
	int maxp = 0;
	int rulerx = 0;
	int rulery = 0;

	cimg_forY(temp, y) {
		int point = 0;
		int tx = 0;
		cimg_forX(temp, x) {
			if (temp(x, y) == 255) {
				point++;
				tx = x;
			}
		}
		if (maxp < point) {
			maxp = point;
			rulery = y;
			rulerx = tx;
		}
	}
	cout << rulerx << " " << rulery << endl;
	temp.display("1");
	int flag = 0;
	int yunx[2];
	for (int i = rulerx; i >= 0; i--) {
		int point = 0;
		for (int j = rulery; j < rulery + 10; j++) {
			if (temp(i, j) == 255) {
				point++;

			}
		}
		if (point > 3) {
			yunx[flag] = i;
			flag++;
			i -= 3;
			if (flag >= 2) break;
		}
	}
	cout << yunx[0] << " " << yunx[1] << endl;
	//temp.display("1");
	
	int tempy = 9;
	vector<int> arrayi;
	for (int i = 0; i < temp.width(); i++) {
		if (temp(i, tempy) == 255) {
			while (true) {
				int point = 0;
				for (int j = tempy; j > tempy - 5; j--) {
					if (temp(i, j) == 255) {
						point++;
					}
				}
				if (point > 3) {
					arrayi.push_back(i);
					i += 5;
				}
				i++;
				if (temp(i, tempy) == 0) {
					arrayi.push_back(-1);
					break;
				}
			}
		}


	}
	int sizt = arrayi.size();
	for (int i = 0; i < sizt; i++) {
		cout << arrayi[i] << " ";
	}
	cout << endl;




	temp.display("1");

	int bili = yunx[0] - yunx[1];
	int sizt_m = arrayi.size();
	vector<double> juli;
	for (int i = 0; i < sizt_m; i++) {
		if (arrayi[i] != -1) {
			double len = yunx[1] - arrayi[i]-2 ;
			len = (len / bili)*0.1;
			juli.push_back(len);
		}
		else {
			juli.push_back(-1);
		}

	}
	int s = 0, e = 0;
	//vector<double> num; 

	temp.display("1");

	double numk[8] = {
		1.00, 1.06, 5.20,5.37,7.31,4.28,1.07,6.28
	};
	int k = 0;
	for (int i = 0; i < juli.size(); i++) {
		if (juli[i] == -1) {
			e = i;
			for (int j = s; j < e - 1; j++) {
				cout << numrecode2[k] << " ";
				cout << setprecision(4) << juli[j] << " " << setprecision(4) << juli[j + 1] << endl;
				k++;
			}
			s = i + 1;
		}
	}
	temp.display("1");
}

void DNum::drawNumImg3() {
	srcImg2.load_bmp("H-Image2k.bmp");
	srcImg2.display("0");
	
	CImg<int> temp;
	temp.resize(srcImg2.width(), srcImg2.height(), 1, 1, 0);
	temp = srcImg2;
	int width = srcImg2.width();
	int x0[4] = { 0,0,0,0 };
	int y0[4] = { 30,1160,1270,1160 };
	int x1[4] = { width,width,width,width };
	int y1[4] = { 140,1270,1350,1350 };
	int color[4] = { 1,3,2,2 };
	for (int i = 0; i < 3; i++) {
		temp = draw_rect(temp, x0[i], y0[i], x1[i], y1[i], color[i]);

	}
	getnewImg(srcImg2, x0[0], y0[0], x1[0], y1[0], "num1.bmp");
	getnewImg(srcImg2, x0[1], y0[1], x1[1], y1[1], "num2.bmp");
	getnewImg(srcImg2, x0[2], y0[2], x1[2], y1[2], "num3.bmp");
	getnewImg(srcImg2, x0[3], y0[3], x1[3], y1[3], "num4.bmp");
	temp.display("drawNumImg");
	srcImg2.save("drawNumImg3.bmp");
}

void DNum::detectNum_3(string filename) {

	Mat srcImage = imread(filename);
	Mat dstImage, grayImage, Image;
	srcImage.copyTo(dstImage);
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	threshold(grayImage, Image, 48, 255, THRESH_BINARY_INV);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(Image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	int i = 0;

	vector<vector<Point>>::iterator It;
	vector<vector<Point>>::iterator temp;
	vector<vector<Point>>::iterator itor2;
	Mat copyImage = Image.clone();
	Rect rect[1000];
	for (It = contours.begin(); It < contours.end(); It++) {

		//画出可包围数字的最小矩形
		int num = (*It).size();
		if (num > 100) {
			continue;
		}
		Point2f vertex[4];
		rect[i] = boundingRect(*It);
		vertex[0] = rect[i].tl();                                                           //矩阵左上角的点
		vertex[1].x = (float)rect[i].tl().x, vertex[1].y = (float)rect[i].br().y;           //矩阵左下方的点
		vertex[2] = rect[i].br();                                                           //矩阵右下角的点
		vertex[3].x = (float)rect[i].br().x, vertex[3].y = (float)rect[i].tl().y;           //矩阵右上方的点

		for (int j = 0; j < 4; j++)
			line(dstImage, vertex[j], vertex[(j + 1) % 4], Scalar(0, 0, 255), 1);

		con[i].x = vertex[0].x;                  //根据中心点判断图像的位置
		con[i].y = (vertex[1].y + vertex[2].y) / 2.0;
		con[i].order = i;
		i++;
	}
	//namedWindow("number", WINDOW_AUTOSIZE);
	//imshow("number", dstImage);
	//sort(con, con + i);
	/*
	for (int t = i - 1, name = 0; t >= 0; t--, name++)
	{
		int k = con[t].order;
		Mat tempImg;
		resize(copyImage(rect[k]), tempImg, Size(30, 30));
		ostringstream fileName;
		fileName << "numberImage/" << name << ".bmp";
		imwrite(fileName.str(), tempImg);
	}
	imwrite("number1.bmp", dstImage);
	*/

	/*
	for (int j = i; j >= 0; j--) {
		int k = con[j].order;
		cout << con[j].order << " " << con[j].x << " " << con[j].y;
		int rows = copyImage(rect[j]).rows;
		int cols = copyImage(rect[j]).cols;
		int same = 0;

		if (cols * rows <=45) {
			con[j].order = -1;
			cout << "delete";
		}
		cout <<  endl;
	}
	*/
	//真正检测函数
	
	int four = 0;
	for (int j = i; j >= 0; j--)
	{

		int k = con[j].order;
		//cout << con[j].order << " " << con[j].x << " " << con[j].y << endl;
		int rows1 = copyImage(rect[k]).rows;
		int cols1 = copyImage(rect[k]).cols;
		if (rows1*cols1 <= 45) {
			//cout << ".";
			continue;
		}
		Mat tempRect = Mat::zeros(Size(30, 30), CV_32FC1);
		resize(copyImage(rect[k]), tempRect, tempRect.size());
		int rows = tempRect.rows;
		int cols = tempRect.cols*tempRect.channels();
		//cvtColor(tempRect, tempRect, COLOR_BGR2GRAY);
		threshold(tempRect, tempRect, 0, 255, THRESH_OTSU);
		tempRect = tempRect.reshape(1, 1);
		Mat inputtemp;
		inputtemp.push_back(tempRect);
		inputtemp.convertTo(inputtemp, CV_32FC1);//更改图片数据的类型，必要，不然会出错

		float r = SVM_params3->predict(inputtemp);   //对所有行进行预测
		cout << r;


		
		//输出匹配的结果
		//cout << j << ":";

		//cout << MatchIndex;
		//deal(tempRect, j + 1);

		//小数点逻辑关系
		//num1

		four++;
		if (four >= 4) {
			four = 0;
			cout << endl;
		}
		else if (four == 1) {
			cout << ".";
		}

	}
	
}

void DNum::trainSVM3() {
	//核心思路：//获取一张图片后会将图片特征写入到容器中，
		//紧接着会将标签写入另一个容器中，这样就保证了特征
		//  和标签是一一对应的关系。
////===============================读取训练数据===============================////
	const int classsum = 11;//图片共有10类，可修改
	const int imagesSum = 10;//每类有张图片，可修改	
	//训了样本图片与测试图片的尺寸应该一样
	const int imageRows = 30;//图片尺寸
	const int imageCols = 30;
	//训练数据，每一行一个训练图片
	Mat trainingData;
	//训练样本标签
	Mat labels;
	//最终的训练样本标签
	Mat clas;
	//最终的训练数据
	Mat traindata;
	//////////////////////从指定文件夹下提取图片//////////////////
	for (int p = 0; p < classsum; p++)//依次提取0到9文件夹中的图片
	{
		oss << "D:/c31/计算机视觉/HW5/hw4/hw4/numberImage/3";
		num += 1;//num从0到9

		int label = num;
		oss << num << "/*.bmp";//图片名字后缀，oss可以结合数字与字符串
		string pattern = oss.str();//oss.str()输出oss字符串，并且赋给pattern
		oss.str("");//每次循环后把oss字符串清空
		vector<Mat> input_images;
		vector<String> input_images_name;
		glob(pattern, input_images_name, false);
		//为false时，仅仅遍历指定文件夹内符合模式的文件，当为true时，会同时遍历指定文件夹的子文件夹
		//此时input_images_name存放符合条件的图片地址
		int all_num = input_images_name.size();
		//文件下总共有几个图片
		//cout << num << ":总共有" << all_num << "个图片待测试" << endl;

		for (int i = 0; i < imagesSum; i++)//依次循环遍历每个文件夹中的图片
		{
			cvtColor(imread(input_images_name[i]), yangben_gray, COLOR_BGR2GRAY);//灰度变换
			threshold(yangben_gray, yangben_thresh, 0, 255, THRESH_OTSU);//二值化
			//循环读取每张图片并且依次放在vector<Mat> input_images内
			input_images.push_back(yangben_thresh);
			dealimage = input_images[i];


			//注意：我们简单粗暴将整个图的所有像素作为了特征，因为我们关注更多的是整个的训练过程
			//，所以选择了最简单的方式完成特征提取工作，除此中外，
			//特征提取的方式有很多，比如LBP，HOG等等
			//我们利用reshape()函数完成特征提取,
			//eshape(1, 1)的结果就是原图像对应的矩阵将被拉伸成一个一行的向量，作为特征向量。 
			dealimage = dealimage.reshape(1, 1);//图片序列化
			trainingData.push_back(dealimage);//序列化后的图片依次存入
			labels.push_back(label);//把每个图片对应的标签依次存入
		}
	}
	//图片数据和标签转变下
	Mat(trainingData).copyTo(traindata);//复制
	traindata.convertTo(traindata, CV_32FC1);//更改图片数据的类型，必要，不然会出错
	Mat(labels).copyTo(clas);//复制


	////===============================创建SVM模型===============================////
	// 创建分类器并设置参数
	SVM_params3 = SVM::create();
	SVM_params3->setType(SVM::C_SVC);//C_SVC用于分类，C_SVR用于回归
	SVM_params3->setKernel(SVM::LINEAR);  //LINEAR线性核函数。SIGMOID为高斯核函数

	SVM_params3->setDegree(0);//核函数中的参数degree,针对多项式核函数;
	SVM_params3->setGamma(1);//核函数中的参数gamma,针对多项式/RBF/SIGMOID核函数; 
	SVM_params3->setCoef0(0);//核函数中的参数,针对多项式/SIGMOID核函数；
	SVM_params3->setC(1);//SVM最优问题参数，设置C-SVC，EPS_SVR和NU_SVR的参数；
	SVM_params3->setNu(0);//SVM最优问题参数，设置NU_SVC， ONE_CLASS 和NU_SVR的参数； 
	SVM_params3->setP(0);//SVM最优问题参数，设置EPS_SVR 中损失函数p的值. 
	//结束条件，即训练1000次或者误差小于0.01结束
	SVM_params3->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01));

	//训练数据和标签的结合
	Ptr<TrainData> tData = TrainData::create(traindata, ROW_SAMPLE, clas);

	// 训练分类器
	SVM_params3->train(tData);//训练

	//保存模型
	//SVM_params->save("C:/Users/zhang/Desktop/opencv——实例/小案例/车牌检测/基于adaboost机器学习/字符识别svm.xml");
	cout << "训练好了！！！" << endl;
}


void DNum::detectNum3_3(string filename)
{
	Mat srcImage = imread(filename);
	Mat dstImage, grayImage, Image;
	srcImage.copyTo(dstImage);
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	threshold(grayImage, Image, 48, 255, THRESH_BINARY_INV);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(Image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	int i = 0;

	vector<vector<Point>>::iterator It;
	vector<vector<Point>>::iterator temp;
	vector<vector<Point>>::iterator itor2;
	Mat copyImage = Image.clone();
	Rect rect[1000];
	for (It = contours.begin(); It < contours.end(); It++) {

		//画出可包围数字的最小矩形
		int num = (*It).size();
		if (num > 100) {
			continue;
		}
		Point2f vertex[4];
		rect[i] = boundingRect(*It);
		vertex[0] = rect[i].tl();                                                           //矩阵左上角的点
		vertex[1].x = (float)rect[i].tl().x, vertex[1].y = (float)rect[i].br().y;           //矩阵左下方的点
		vertex[2] = rect[i].br();                                                           //矩阵右下角的点
		vertex[3].x = (float)rect[i].br().x, vertex[3].y = (float)rect[i].tl().y;           //矩阵右上方的点

		for (int j = 0; j < 4; j++)
			line(dstImage, vertex[j], vertex[(j + 1) % 4], Scalar(0, 0, 255), 1);

		con[i].x = vertex[0].x;                  //根据中心点判断图像的位置
		con[i].y = vertex[1].y;
		con[i].order = i;
		i++;
	}
	//imwrite("number2.bmp", dstImage);
	//真正检测函数
	string tempNum = "";
	int four = 0;
	for (int j = i - 1; j >= 0; j--)
	{

		//调整矩阵 与数据库图片大小一致
		int k = con[j].order;
		//cout << con[j].order << " " << con[j].x << " " << con[j].y << endl;
		int rows1 = copyImage(rect[k]).rows;
		int cols1 = copyImage(rect[k]).cols;
		if (rows1*cols1 <= 45) {
			//cout << ".";
			continue;
		}
		Mat tempRect = Mat::zeros(Size(30, 30), CV_32FC1);
		resize(copyImage(rect[k]), tempRect, tempRect.size());
		int rows = tempRect.rows;
		int cols = tempRect.cols*tempRect.channels();
		//cvtColor(tempRect, tempRect, COLOR_BGR2GRAY);
		threshold(tempRect, tempRect, 0, 255, THRESH_OTSU);
		tempRect = tempRect.reshape(1, 1);
		Mat inputtemp;
		inputtemp.push_back(tempRect);
		inputtemp.convertTo(inputtemp, CV_32FC1);//更改图片数据的类型，必要，不然会出错

		float r = SVM_params3->predict(inputtemp);   //对所有行进行预测
		cout << r;
		tempNum += int(r) + '0';

		//小数点逻辑关系
		//num1

		four++;
		if (four >= 3) {
			four = 0;
			cout << endl;
			numrecode3.push_back(tempNum);
			tempNum = "";
		}
		else if (four == 1) {
			cout << ".";
			tempNum += ".";
		}

	}

}

void DNum::dectpoint() {
	CImg<int> temp;
	temp.load_bmp("num2.bmp");
	temp.display();
	cimg_forXY(temp, x, y) {
		if (x >= 70) {
			temp(x, y, 0) = 255;
			temp(x, y, 1) = 255;
			temp(x, y, 2) = 255;
		}
	}
	vector<int> key;
	cimg_forY(temp, y) {
		int point = 0;
		cimg_forX(temp, x) {
			if (temp(x, y) == 0) {
				point++;
			}
		}
		if (point > 50) {


			//point = 0;
			key.push_back(y);
		}
	}
	int len = key.size();
	cout << len << endl;
	for (int i = 0; i < key.size(); i++) {
		cimg_forX(temp, x) {
			if (temp(x, key[i]) == 0) {
				temp(x, key[i],0) = 255;
				temp(x, key[i],1) = 255;
				temp(x, key[i],2) = 255;
			}
		}
	}
	temp.display();
	temp.save("ex_num2.bmp");
}

void DNum::detectNum2_3(string filename) {
	Mat srcImage = imread(filename);
	Mat dstImage, grayImage, Image;
	srcImage.copyTo(dstImage);
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	threshold(grayImage, Image, 48, 255, THRESH_BINARY_INV);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(Image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	int i = 0;

	vector<vector<Point>>::iterator It;
	vector<vector<Point>>::iterator temp;
	vector<vector<Point>>::iterator itor2;
	Mat copyImage = Image.clone();
	Rect rect[1000];
	for (It = contours.begin(); It < contours.end(); It++) {

		//画出可包围数字的最小矩形
		int num = (*It).size();
		if (num > 100) {
			continue;
		}
		Point2f vertex[4];
		rect[i] = boundingRect(*It);
		vertex[0] = rect[i].tl();                                                           //矩阵左上角的点
		vertex[1].x = (float)rect[i].tl().x, vertex[1].y = (float)rect[i].br().y;           //矩阵左下方的点
		vertex[2] = rect[i].br();                                                           //矩阵右下角的点
		vertex[3].x = (float)rect[i].br().x, vertex[3].y = (float)rect[i].tl().y;           //矩阵右上方的点

		for (int j = 0; j < 4; j++)
			line(dstImage, vertex[j], vertex[(j + 1) % 4], Scalar(0, 0, 255), 1);

		con[i].x = vertex[0].x;                  //根据中心点判断图像的位置
		con[i].y = vertex[1].y;
		con[i].order = i;
		i++;
	}
	imwrite("number3.bmp", dstImage);
	sort(con, con + i);
	//真正检测函数

	int four = 0;
	for (int j = 0; j < i; j++)
	{

		//调整矩阵 与数据库图片大小一致
		int k = con[j].order;
		//cout << con[j].order << " " << con[j].x << " " << con[j].y << endl;
		int rows1 = copyImage(rect[k]).rows;
		int cols1 = copyImage(rect[k]).cols;
		Mat tempRect = Mat::zeros(Size(30, 30), CV_32FC1);
		resize(copyImage(rect[k]), tempRect, tempRect.size());
		int rows = tempRect.rows;
		int cols = tempRect.cols*tempRect.channels();
		//cvtColor(tempRect, tempRect, COLOR_BGR2GRAY);
		threshold(tempRect, tempRect, 0, 255, THRESH_OTSU);
		tempRect = tempRect.reshape(1, 1);
		Mat inputtemp;
		inputtemp.push_back(tempRect);
		inputtemp.convertTo(inputtemp, CV_32FC1);//更改图片数据的类型，必要，不然会出错

		float r = SVM_params3->predict(inputtemp);   //对所有行进行预测
		if (r == 10) {
			cout << ".";
			//tempNum += ".";
		}
		else {
			cout << r;
			//tempNum += int(r) + '0';
		}





		four++;
		if (four >= 3) {
			four = 0;
			cout << endl;
		}
	}
}
void DNum::drawRuler3()
{
	CImg<int> temp;
	temp.load_bmp("num4.bmp");
	temp.display();
	vector<int> arrayi;
	int flag = 0;
	cimg_forX(temp, x) {
		if (temp(x, 10) == 0) {
			arrayi.push_back(x);
			flag++;
			x += 5;
		}
		if (flag >= 2) {
			arrayi.push_back(-1);
			flag = 0;
		}
	}
	for (int i = 0; i < arrayi.size(); i++) {
		cout << arrayi[i] << " ";
	}
	cout << endl;
	int yun[2] = { 2129,2108 };
	int bili = yun[0] - yun[1];
	vector<double> juli;
	for (int i = 0; i < arrayi.size(); i++) {
		if (arrayi[i] != -1) {
			double len = yun[0] - arrayi[i];
			len = double(len / bili)*0.1;
			juli.push_back(len);
		}
		else {
			juli.push_back(-1);
		}
	}
	temp.display();
	double numk[8] = {
	1.00, 1.06, 5.20,5.37,7.31,4.28,1.07,6.28
	};
	int e = 0,s = 0;
	
	int k = 0;
	for (int i = 0; i < juli.size(); i++) {
		if (juli[i] == -1) {
			e = i;
			for (int j = s; j < e - 1; j++) {
				cout << numrecode3[k] << " ";
				cout << setprecision(4) << juli[j] << " " << setprecision(4) << juli[j + 1] << endl;
				k++;
			}
			s = i + 1;
		}
	}
	temp.display("1");
}


double getjuli(int x,int x0,CImg<int> ruler) {

}