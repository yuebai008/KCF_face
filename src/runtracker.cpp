#include <iostream>
#ifdef _WIN32
#pragma once
#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) \
  CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif //_DEBUG

#pragma comment( lib, cvLIB("core") )
#pragma comment( lib, cvLIB("imgproc") )
#pragma comment( lib, cvLIB("highgui") )

#endif //_WIN32

#if defined(__unix__) || defined(__APPLE__)

#ifndef fopen_s

#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL
#endif //fopen_s

#endif //__unix

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "face_identification.h"
#include "recognizer.h"
#include "face_detection.h"
#include "face_alignment.h"

#include "math_functions.h"

#include "kcftracker.hpp"

#include <dirent.h>

using namespace std;
using namespace cv;
using namespace seeta;

#define TEST(major, minor) major##_##minor##_Tester()
#define EXPECT_NE(a, b) if ((a) == (b)) std::cout << "ERROR: "
#define EXPECT_EQ(a, b) if ((a) != (b)) std::cout << "ERROR: "

#ifdef _WIN32
//std::string DATA_DIR = "../../data/";
std::string MODEL_DIR = "../model/";
#else
//std::string DATA_DIR = "./data/";
std::string MODEL_DIR = "../model/";
#endif

cv::Rect box;
cv::Rect face_box;
bool drawing_box = false;
bool gotBB = false;
bool tl = true;
bool rep = false;
bool fromfile=false;
ofstream outfile("detecting_and_tracking_time.txt");

void face_identity()
{

}

void mouseHandler(int event, int x, int y, int flags, void *param){  //using mouse select box
  switch( event ){
  case CV_EVENT_MOUSEMOVE:
    if (drawing_box){
        box.width = x-box.x;
        box.height = y-box.y;
    }
    break;
  case CV_EVENT_LBUTTONDOWN:
    drawing_box = true;
    box = cv::Rect( x, y, 0, 0 );
    break;
  case CV_EVENT_LBUTTONUP:
    drawing_box = false;
    if( box.width < 0 ){
        box.x += box.width;
        box.width *= -1;
    }
    if( box.height < 0 ){
        box.y += box.height;
        box.height *= -1;
    }
    gotBB = true;
    break;
  }
}
int main(int argc, char* argv[]){

	VideoCapture capture;
	capture.open(0);

	if(!capture.isOpened())
	{
		cout<< "capture device failed to open!"<<endl;
		return 1;
	}

	if (argc > 5) return -1;

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;

	for(int i = 0; i < argc; i++){
		if ( strcmp (argv[i], "hog") == 0 )
			HOG = true;
		if ( strcmp (argv[i], "fixed_window") == 0 )
			FIXEDWINDOW = true;
		if ( strcmp (argv[i], "singlescale") == 0 )
			MULTISCALE = false;
		if ( strcmp (argv[i], "show") == 0 )
			SILENT = false;
		if ( strcmp (argv[i], "lab") == 0 ){
			LAB = true;
			HOG = true;
		}
		if ( strcmp (argv[i], "gray") == 0 )
			HOG = false;
	}
	
	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
     //seetaface_init
      // Initialize face detection model 
    seeta::FaceDetection detector("./seeta_fd_frontal_v1.0.bin");
    detector.SetMinFaceSize(40);
    detector.SetScoreThresh(2.f);
    detector.SetImagePyramidScaleFactor(0.8f);
    detector.SetWindowStep(4, 4);

    // Initialize face alignment model 
    seeta::FaceAlignment point_detector("./seeta_fa_v1.1.bin");
    // Initialize face Identification model 
    FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());
	// Frame readed
	//Mat frame;

	// Tracker results
	cv::Rect result;

	// Write Results
	ofstream resultsFile;
	string resultsPath = "output.txt";
	resultsFile.open(resultsPath);

	

  cvNamedWindow("KCF",CV_WINDOW_AUTOSIZE);
  //cvSetMouseCallback( "KCF", mouseHandler, NULL );
  
  //Read parameters file
  Mat frame;
  Mat last_face_box_color;
  cv::Mat face_box_color;
  ImageData face_box_data_color;
  float gallery_fea[2048];
  int first_face_flag=0;

  capture.set(CV_CAP_PROP_FRAME_WIDTH,640);
  capture.set(CV_CAP_PROP_FRAME_HEIGHT,360);

GETBOUNDINGBOX:
    /*while(!gotBB)
    {
      if (!fromfile){
        capture >> frame;
      }
      else
        first.copyTo(frame);
      //cvtColor(frame, last_gray, CV_RGB2GRAY);
      rectangle( frame, Point( box.x, box.y ), Point( box.x+box.width, box.y+box.height), Scalar( 0, 255, 255 ), 1, 8 );
      imshow("KCF", frame);
      if (cvWaitKey(33) == 'q')
	      return 0;
    }
    cvSetMouseCallback( "KCF", NULL, NULL );
    printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);
*/

  //int frame_flag = 0;//5 frame seetaface_detect
  
  while(!gotBB)
  {
    capture >> frame;

 
    cv::Mat img_gray;

    if (frame.channels() != 1)
      cv::cvtColor(frame, img_gray, cv::COLOR_BGR2GRAY);
    else
      img_gray = frame;

    seeta::ImageData img_data;
    img_data.data = img_gray.data;
    img_data.width = img_gray.cols;
    img_data.height = img_gray.rows;
    img_data.num_channels = 1;

    long t0 = cv::getTickCount();
    std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);


    int32_t num_face = static_cast<int32_t>(faces.size()); // detect face

    int32_t i=0;
    int face_size=0;
    float max_sim = 0;

    for (int32_t i = 0; i < num_face; i++) {
    //if(i<num_face){
      box.x = faces[i].bbox.x;
      box.y = faces[i].bbox.y;
      box.width = faces[i].bbox.width;
      box.height = faces[i].bbox.height;
      //first detect face
      if((box.x+box.width)>620||(box.x<20)||box.y<20||(box.y+box.height)>460)
        continue;
      if(box.width*box.height > face_size && first_face_flag == 0)
      {
          face_size=box.width*box.height;
          face_box=box;
          gotBB=true;
          face_box_color = frame(face_box).clone();
          cv::rectangle(frame, face_box, CV_RGB(0, 0, 255), 4, 8, 0);
      }

      if(first_face_flag != 0){
          outfile << "The first_face_flag is " << first_face_flag <<'\n';

        cv::Mat src_img = frame(box);
        Mat probe_img_color;
       // probe_img_color=src_img;
        //if(src_img.cols>200)
        cv::resize(src_img,probe_img_color,Size(100,100));





        cv::Mat probe_img_gray;
        cv::cvtColor(probe_img_color, probe_img_gray, CV_BGR2GRAY);

        ImageData probe_img_data_color(probe_img_color.cols, probe_img_color.rows, probe_img_color.channels());
        probe_img_data_color.data = probe_img_color.data;

        ImageData probe_img_data_gray(probe_img_gray.cols, probe_img_gray.rows, probe_img_gray.channels());
        probe_img_data_gray.data = probe_img_gray.data;

        std::vector<seeta::FaceInfo> probe_faces = detector.Detect(probe_img_data_gray);
        int32_t probe_face_num = static_cast<int32_t>(probe_faces.size());

        if(!probe_face_num) {
          std::cout << "Faces are not detected.";
          continue;
        }

        seeta::FacialLandmark probe_points[5];
        point_detector.PointDetectLandmarks(probe_img_data_gray, probe_faces[0], probe_points);
        /*int32_t probe_points_num = static_cast<int32_t>(probe_points.size());
        if(!probe_points_num) continue;*/
      /* for (int i = 0; i<5; i++)
        {
          //cv::circle(gallery_img_color, cv::Point(gallery_points[i].x, gallery_points[i].y), 2,
           // CV_RGB(0, 255, 0));
          cv::circle(probe_img_color, cv::Point(probe_points[i].x, probe_points[i].y), 2,
            CV_RGB(0, 255, 0));
        }
       // cv::imwrite("gallery_point_result.jpg", gallery_img_color);
        cv::imwrite("probe_point_result.jpg", probe_img_color);*/

        // Extract face identity feature
        float probe_fea[2048];

        face_recognizer.ExtractFeatureWithCrop(probe_img_data_color, probe_points, probe_fea);

        // Caculate similarity of two faces
        float sim_cal = face_recognizer.CalcSimilarity(gallery_fea, probe_fea);

        std::cout << sim_cal <<endl;

        outfile << "The sim_cal is " << sim_cal <<'\n';
        if(sim_cal>max_sim && sim_cal>0.5)
        {
          max_sim = sim_cal;
          face_box = cv::Rect(box.x,box.y,box.width,box.height);
         // std::cout << sim_cal <<endl;
          gotBB=true;
          face_box_color = frame(face_box).clone();
          cv::rectangle(frame, face_box, CV_RGB(0, 0, 255), 4, 8, 0);
          std::cout << "gotBB=true" <<endl;
        }
      }
    }
         
      
    //seetaface_align_ident
    cout <<"The first_face_flag is "<<first_face_flag<<endl;



    /*if (!num_face) {
      gotBB = false;
      //frame_flag=0;
    }*/

    long t1 = cv::getTickCount();
    double secs = (t1 - t0)/cv::getTickFrequency();
    //output detect time and frame size 
    cout << endl << "The detection takes " << secs << " seconds!" << endl << endl;
    cout << "The frame size is width: " << frame.cols << "  height: " << frame.rows << endl << endl;
    outfile << "The detection takes " << secs << " seconds!"<<'\n';
    outfile << "The frame size is width: " << frame.cols << "  height: " << frame.rows<<'\n';



    imshow("KCF", frame);
    if (cvWaitKey(10) == 'q')
	    return 0;
  }

//	while ( getline(listFramesFile, frameName) ){
    // Frame counter
	int nFrames = 0;
 REPEAT:
	while (capture.read(frame)){
		//frameName = frameName;

		// Read each frame from the list
		//frame = imread(frameName, CV_LOAD_IMAGE_COLOR);
       // capture>>frame;
		//cvtColor(frame, current_gray, CV_RGB2GRAY);
		// First frame, give the groundtruth to the tracker
		if (nFrames == 0) {
      //
			//tracker.init( Rect(xMin, yMin, width, height), frame );
			tracker.init( face_box, frame );
			//rectangle( frame, Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar( 0, 255, 255 ), 1, 8 );
			resultsFile << face_box.x << "," << face_box.y << "," << face_box.width << "," << face_box.height << endl;
		}
		// Update
		else{
			long t0 = cv::getTickCount();
			result = tracker.update(frame);
			long t1 = cv::getTickCount();
			double secs1 = (t1 - t0)/cv::getTickFrequency();
			cout << endl << "The tracking takes " << secs1 << " seconds!" << endl;
			outfile << "The tracking takes " << secs1 << " seconds!"<<'\n';
			rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 255, 255 ), 1, 8 );
			resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
		}

    imshow("KCF",frame);
    /*if(((result.x+result.width)>620) || (result.x<20) || (result.y<20) || ((result.y+result.height)>460))
    {
      //imshow("KCF",frame);
      //capture>>frame;
     // if(cvWaitKey(20)=='q')
       // break;
      goto REPEAT;
    }*/

		nFrames++;
		outfile<< "nFrames"<<nFrames<<'\n';

		if(nFrames>30)
		{
        last_face_box_color=face_box_color;
        cv::Mat gallery_img_color;
        //if(last_face_box_color.cols>200) 
          resize(last_face_box_color,gallery_img_color,Size(100,100));
       // else2
         // gallery_img_color=last_face_box_color;
        cv::Mat gallery_img_gray;
        cv::cvtColor(gallery_img_color, gallery_img_gray, CV_BGR2GRAY);

        ImageData gallery_img_data_color(gallery_img_color.cols, gallery_img_color.rows, gallery_img_color.channels());
        gallery_img_data_color.data = gallery_img_color.data;

        ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
        gallery_img_data_gray.data = gallery_img_gray.data;

                // Detect faces
        std::vector<seeta::FaceInfo> gallery_faces = detector.Detect(gallery_img_data_gray);
        int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());
        if(!gallery_face_num&&nFrames>0) {
          std::cout << "gallery_Faces are not detected.";
          nFrames-=1;
          goto REPEAT;
        }

                // Detect 5 facial landmarks
        seeta::FacialLandmark gallery_points[5];
        point_detector.PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);
        face_recognizer.ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, gallery_fea);

      first_face_flag++;
      if(first_face_flag == 300)
        first_face_flag=0;
			gotBB =false;
			goto GETBOUNDINGBOX;
		}
		
    //capture>>frame;
		if(cvWaitKey(10)=='q')
			break;
		/*if (!SILENT){
			imshow("Image", frame);
			waitKey(33);
		}*/
	}
	resultsFile.close();
  outfile.close();
	//listFile.close();

}
