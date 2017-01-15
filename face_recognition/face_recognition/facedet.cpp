#include "cv.h"
#include <stdio.h>
#include "facedet.h"
#include "highgui.h"

// File-level variables
CvHaarClassifierCascade * pCascade = 0;  // the face detector
CvHaarClassifierCascade * eCascade = 0;  // the eye detector
CvMemStorage * pStorage = 0;             // memory for face detector to use
CvMemStorage * eStorage = 0;             // memory for eye detector to use
CvSeq * pFaceRectSeq;                    // face memory-access interface
CvSeq * eFaceRectSeq;                    // eye memory-access interface

//function prototypes
int eyes_detect_normalize( IplImage* img2,IplImage* imgdst);

//////////////////////////////////
// initFaceDet()
//
int initPathDet(const char * haarCascadePath,const char * eyeCascadePath)
{
	if( !((pStorage = cvCreateMemStorage(0)) && (eStorage = cvCreateMemStorage(0))) )
	{
		fprintf(stderr, "Can\'t allocate memory for face detection\n");
		return 0;
	}

	pCascade = (CvHaarClassifierCascade *)cvLoad( haarCascadePath, 0, 0, 0 );
    eCascade = (CvHaarClassifierCascade *)cvLoad(  eyeCascadePath, 0, 0, 0 );
	
	if( !(pCascade&&eCascade))
	{
		fprintf(stderr, "Can\'t load Haar classifier cascade from\n"
		                "Please check that this is the correct path\n"
						);
		return 0;
	}

	return 1;
}


//////////////////////////////////
// closeFaceDet()
//
void closeFaceDet()
{
	if(pCascade) 
	{
		      cvReleaseHaarClassifierCascade(&pCascade);
	          cvReleaseHaarClassifierCascade(&eCascade);
	}
	if(pStorage)
	{
			 cvReleaseMemStorage(&pStorage);
             cvReleaseMemStorage(&eStorage);
	}
}


//////////////////////////////////
// detectFace()
//
CvRect* face_detect( IplImage* img,IplImage* imgdst2 )
{

	double t2=(double)cvGetTickCount();
    int minFaceSize=img->width/5,scale1=2;
    CvRect* face_rect=0;
	int eye=0;

  /* IplImage* small_img = cvCreateImage( cvSize( cvRound (img->width/scale1),
                         cvRound (img->height/scale1)), 8, 1 );


    cvResize( img, small_img, CV_INTER_LINEAR );
    cvEqualizeHist( small_img, small_img );*/

    if(pCascade)
    {

        CvSeq* faces = cvHaarDetectObjects( img, pCascade, pStorage,
                                            1.1, 6, 0/*CV_HAAR_DO_CANNY_PRUNING*/,
                                            cvSize(minFaceSize, minFaceSize) );
        if(faces && faces->total )
	   {

			face_rect = (CvRect*)cvGetSeqElem(faces,0);
            
			if(!face_rect) return face_rect;
       
			//另存偵測到的人臉小影像
			IplImage* imgt = cvCreateImage(cvSize(scale1*(face_rect->width),scale1*(face_rect->height)), IPL_DEPTH_8U, 1 );
			cvSetImageROI(img, *face_rect); 
            cvResize(img, imgt , CV_INTER_AREA );
            cvEqualizeHist(imgt,imgt);
            
			/*cvNamedWindow("face detection result",1);	
			cvShowImage("face detection result",imgt);
			cvWaitKey(0);
			cvDestroyWindow("face detection result");*/

			t2=(double)cvGetTickCount()-t2;
			printf("\nface detect time   =   %gms\n",t2/((double)cvGetTickFrequency()*1000.));
			//執行眼睛偵測並取出正規化後的人臉影像
			eye=eyes_detect_normalize(imgt,imgdst2);
			cvReleaseImage( &imgt );
		}
	}
    if(eye)return face_rect;
	else return NULL;
}

int eyes_detect_normalize( IplImage* img2,IplImage* imgdst)
{
    int n;
    int scale2 = 2;
    CvRect* eye_rect =0;

	double t3=(double)cvGetTickCount();
 

    IplImage* small_img2 = cvCreateImage( cvSize( cvRound (img2->width/scale2),
                         cvRound (img2->height/scale2)), 8, 1 );


    cvResize( img2, small_img2, CV_INTER_LINEAR );
    cvEqualizeHist( small_img2, small_img2 );

    if( eCascade)
	{

		CvSeq* eyes = cvHaarDetectObjects(  small_img2, eCascade, eStorage,
                                            1.1, 2, 0/*CV_HAAR_DO_CANNY_PRUNING*/,
                                            cvSize(22, 5) );
         
		int height3=img2->height;
		int width3=img2->width;
		int channels3=img2->nChannels;
		int step3=img2->widthStep/sizeof(uchar);
		uchar * data3=(uchar*)img2->imageData;

		//faces 儲存每張偵測到人臉的 x,y,width,height 資訊
		for( n = 0; n < (eyes ? eyes->total : 0); n++ )
        //if(eyes && eyes->total )
		{

		    eye_rect = (CvRect*)cvGetSeqElem( eyes, n);
			
			//另存眼睛偵測後的人臉   
			//--------------- 調整所要取出的小影像大小 ---------------

			IplImage* imgt2 = cvCreateImage( cvSize( 1.2*eye_rect->width*scale2 , 4.7*eye_rect->height*scale2 ), IPL_DEPTH_8U, 1 );
			int height4=imgt2->height;
			int width4=imgt2->width;
			int channels4=imgt2->nChannels;
			int step4=imgt2->widthStep/sizeof(uchar);
			uchar * data4=(uchar*)imgt2->imageData;


			//--------------- 調整所要取出的小影像位置 ---------------
			int eye_rtx=eye_rect->x*scale2 - 0.1*eye_rect->width*scale2;
			int eye_rty=eye_rect->y*scale2 - 0.7*eye_rect->height*scale2;

			int k1,k2;
			for(k1=0;k1<height4;k1++)//
			{
				for(k2=0;k2<width4;k2++)//
				{
					data4[k1*step4+k2*channels4] = data3[(eye_rty+k1)*step3+(eye_rtx+k2)*channels3];

				}
			}
           /* cvNamedWindow("eye detection",1);	
			cvShowImage("eye detection",imgt2);
			cvWaitKey(0);
			cvDestroyWindow("eye detection");*/
			
			t3=(double)cvGetTickCount()-t3;
			printf(" eye detect time   =   %gms\n",t3/((double)cvGetTickFrequency()*1000.));
			//將矩陳的影像縮放以做正規化			

			IplImage* imgt3 = cvCreateImage( cvSize( 112 , 96 ), IPL_DEPTH_8U, 1 );
 
			double t4=(double)cvGetTickCount();
			cvResize( imgt2 , imgt3 , CV_INTER_LINEAR );
			t4=(double)cvGetTickCount()-t4;
			printf("image normalize time = %gms\n",t4/((double)cvGetTickFrequency()*1000.));
			/*cvNamedWindow("image normalize",1);	
			cvShowImage("image normalize",imgt3);
			cvWaitKey(0);
			cvDestroyWindow("image normalize");*/
			cvCopy(imgt3,imgdst);
		}

	}	 
	 if(eye_rect) return 1;
	 else return 0;
     cvReleaseImage( &small_img2 );
}

