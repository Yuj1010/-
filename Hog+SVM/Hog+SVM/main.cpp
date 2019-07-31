#include <iostream>
#include <fstream>
#include <opencv2/core/bufferpool.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
	//creat SVM classfier
	Ptr<SVM> svm = SVM::create();
	//load train file
	svm = SVM::load<SVM>("KTH_HOG6.xml");
	int count = 0;
	if (!svm)
	{
		cout << "Load file failed..." << endl;
	}
	
	
	Mat test;

	test = imread("F:\\predict\\2\\JPEG\\2_19.jpg");
	

	cout << "Load test image done..." << endl;


	
	//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	HOGDescriptor hog(Size(128, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9);

	vector<float> descriptors;//HOG����������
	hog.compute(test, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)

	int r = svm->predict(descriptors);   //�������н���Ԥ��
	switch (r)
	{
	case 0:
		cout << "The action is boxing" << endl;
		break;
	case 1:
		cout << "The action is handclapping" << endl;
		break;
	case 2:
		cout << "The action is handwaving" << endl;
		break;
	case 3:
		cout << "The action is walking" << endl;
		break;
	case 4:
		cout << "The action is running" << endl;
		break;
	}
	imshow("org", test);
	waitKey(60000);
	
	//system("pause");
	return 1;
}