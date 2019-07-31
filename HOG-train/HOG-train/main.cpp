#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
	//winsize(64,128),blocksize(16,16),blockstep(8,8),cellsize(8,8),bins9
	//检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	HOGDescriptor hog(Size(128, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9);

	//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定

	Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
	Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；

	Ptr<SVM> svm = SVM::create();//SVM分类器

	string ImgName;//图片名(绝对路径)

	ifstream finPos("Image/img.txt");//正样本图片的文件名列表

	if (!finPos)
	{
		cout << "Pos/Neg imglist reading failed..." << endl;
		return 1;
	}

	for (int num = 0; num < 250 && getline(finPos, ImgName); num++)
	{
		std::cout << "Now processing original positive image: " << ImgName << endl;
		ImgName = "Image/" + ImgName;//加上正样本的路径名

		Mat src = imread(ImgName);//读取图片

		//if (CENTRAL_CROP)
		//	src = src(Rect(16, 16, 128, 128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素

		vector<float> descriptors;//HOG描述子向量
		hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)

		//处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
		if (0 == num)
		{
			DescriptorDim = descriptors.size();
			//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
			sampleFeatureMat = Mat::zeros(250, DescriptorDim, CV_32FC1);
			//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
			sampleLabelMat = Mat::zeros(250, 1, CV_32SC1);//sampleLabelMat的数据类型必须为有符号整数型
		}

		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
		for (int i = 0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(num, i) = descriptors[i];//第num个样本的特征向量中的第i个元素

		sampleLabelMat.at<int>(num, 0) = num / 50;
	}

	finPos.close();


	//输出样本的HOG特征向量矩阵到文件
	svm->setType(SVM::C_SVC);
	svm->setC(0.01);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 3000, 1e-6));

	std::cout << "Starting training..." << endl;
	svm->train(sampleFeatureMat, ROW_SAMPLE, sampleLabelMat);//训练分类器
	std::cout << "Finishing training..." << endl;
	//将训练好的SVM模型保存为xml文件
	svm->SVM::save("KTH_HOG6.xml");

	//imshow("src", src);

	waitKey();
	return 0;

}
