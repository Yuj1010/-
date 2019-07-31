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
	//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	HOGDescriptor hog(Size(128, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9);

	//HOG���������������HOG�����ӵ�
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������

	Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��
	Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��

	Ptr<SVM> svm = SVM::create();//SVM������

	string ImgName;//ͼƬ��(����·��)

	ifstream finPos("Image/img.txt");//������ͼƬ���ļ����б�

	if (!finPos)
	{
		cout << "Pos/Neg imglist reading failed..." << endl;
		return 1;
	}

	for (int num = 0; num < 250 && getline(finPos, ImgName); num++)
	{
		std::cout << "Now processing original positive image: " << ImgName << endl;
		ImgName = "Image/" + ImgName;//������������·����

		Mat src = imread(ImgName);//��ȡͼƬ

		//if (CENTRAL_CROP)
		//	src = src(Rect(16, 16, 128, 128));//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������

		vector<float> descriptors;//HOG����������
		hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)

		//�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������
		if (0 == num)
		{
			DescriptorDim = descriptors.size();
			//��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat
			sampleFeatureMat = Mat::zeros(250, DescriptorDim, CV_32FC1);
			//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
			sampleLabelMat = Mat::zeros(250, 1, CV_32SC1);//sampleLabelMat���������ͱ���Ϊ�з���������
		}

		//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
		for (int i = 0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(num, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��

		sampleLabelMat.at<int>(num, 0) = num / 50;
	}

	finPos.close();


	//���������HOG�������������ļ�
	svm->setType(SVM::C_SVC);
	svm->setC(0.01);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 3000, 1e-6));

	std::cout << "Starting training..." << endl;
	svm->train(sampleFeatureMat, ROW_SAMPLE, sampleLabelMat);//ѵ��������
	std::cout << "Finishing training..." << endl;
	//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�
	svm->SVM::save("KTH_HOG6.xml");

	//imshow("src", src);

	waitKey();
	return 0;

}
