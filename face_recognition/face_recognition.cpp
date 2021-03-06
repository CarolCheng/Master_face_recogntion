#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <opencv\highgui.h>
#include <opencv\cv.h>
#include <opencv\cvaux.h>
#include "facedet.h"

#define		TPL_WIDTH		112
#define		TPL_HEIGHT		96
#define		SEARCH_RANGE_X	10
#define		SEARCH_RANGE_Y	10
#define		THRESHOLD		0.5
int			object_x0		= 0;
int			object_y0		= 0;
int			object_x0_ini	= 0;
int			object_y0_ini	= 0;

//// Global variables
IplImage ** pFaceImg         = 0;		// array of images
IplImage ** faceImgArr        = 0;		// array of face images
IplImage * pAvgTrainImg       = 0;		// the average image
IplImage ** eigenVectArr      = 0;		// eigenvectors
CvCapture *video=0;      
CvMat    *  personNumTruthMat = 0;		// array of person numbers
CvMat    *  personName = 0;				// array of person name
CvMat * eigenValMat           = 0;		// eigenvalues
CvMat * projectedTrainFaceMat = 0;		// projected training faces
CvMat *projectedaverage=0;                // median  of projected training faces of each people
CvMat *projectmedian=0;                   // median  of projected training faces of all people
CvMat *thres_eachdata=0;
const char *faceCascadePath="haarcascade_frontalface_alt2.xml";		//face-detection xml file
const char *eyeCascadePath="parojos.xml";							//eye-detection xml file
int nWebCam=10;							//the number of video frame in order to increase face recognition correct rate
int nTrainFaces               = 0;		// the number of training images
int nEigens                   = 0;		// the number of eigenvalues
double thre_alldata;

//// Function prototypes
void recognize(int nTestFaces,int sel);
void doPCA();
void closeFaceDet();
void capturevideoframe();
void  decideredsult(int *idnumber,CvMat *trainPersonNumMat,CvMat *trainpersonName,int sel);
void exitprogram();
int loadTrainingData(CvMat ** pTrainPersonNumMat,CvMat **personName,CvMat **trainaverage);
int loadFaceImgArray(char * filename);
int  findNearestNeighbor(CvMat *trainPersonNumMat,float * projectedTestFace);
int  loadFaceImgArray(char * filename);
int recog_stranger(float * projectedTestFace);
int threshold(float * projectedTestFace,int iNearest);
int addsum(int a);
int initPathDet(const char * haarCascadePath,const char * eyeCascadePath);
CvRect* face_detect( IplImage* img,IplImage* imgdst2 );
IplImage* track_object(IplImage	*frame,IplImage	*tpl,IplImage *match_res,CvRect *face);
void quick_sort(double *A,int *B,int left,int right,int k);
void swap(double *a,double *b);
void intswap(int *a,int *b);
CvMat * trainPersonNumMat = 0;		 // the person numbers during training
CvMat * trainpersonName = 0;		 // the person name during training

//////////////////////////////////
// main()
//
void main( int argc, char** argv )
{
	//進行變數的初始化
	int sel=0;
	int nTestFaces=0;
	char keyin[10],filename[20]="test.txt";

	sel=initPathDet(faceCascadePath,eyeCascadePath);
    // load the saved training data
	if( !loadTrainingData( &trainPersonNumMat,&trainpersonName,&projectedaverage) ) return;
	
	if(sel==1)
	{

		printf("face recognition system:\n");
		printf("1.read test.txt\n"
		       "2.capture image sequence from avi\n"
			   "3.capture image sequence from camera\n"
		       "Any number: End\n");
		scanf("%d",&sel);

		if(sel==1)
		{
			nTestFaces=loadFaceImgArray(filename);
			double t6=(double)cvGetTickCount();
			recognize(nTestFaces,sel);
			t6=(double)cvGetTickCount()-t6;    
			printf("Program execution total time = %gms\n",t6/((double)cvGetTickFrequency()*1000.));
	    }
	    else 
		{
			switch (sel)
			{
			case 2:
				/*/printf("以下為要辨識的影像,請輸入你要辨識哪一個\n"
					"主人1號: hoo.avi\n"
					"主人2號: zoo.avi\n"
					"主人3號: carol.avi\n"
					"其他: stranger.avi\n");
				scanf("%s",&sel);*/
				video= cvCaptureFromFile("stranger.avi");
				break;
			case 3:
				video= cvCreateCameraCapture(0);
				break;
			default:
				printf("沒有這個選項\n");
				exit(0);
			}

			printf( "\n********************************************\n"
					   "Face recognition in video sequence"
					 "\n********************************************\n" );

			if(!video)
			{
				printf("無法從攝影機取影像");
				exit(0);
			}  
		   //
			
			cvNamedWindow("Video",CV_WINDOW_AUTOSIZE); 

			while(sel!=0)
			{
			     
				 double t6=(double)cvGetTickCount();
				 capturevideoframe();
				 recognize(nWebCam,sel);
				 t6=(double)cvGetTickCount()-t6;

				 printf("Program execution total time = %gms\n\n",t6/((double)cvGetTickFrequency()*1000.));
				 printf("If you want to stop doing face recognition,you input '0' :");
				 scanf("%d",&sel);

			}
			cvDestroyWindow("Video");
		}
		exitprogram();
	}
    
}
//////////////////////////////////
// recognize()
//
void recognize(int nTestFaces,int sel)   // nTestFaces:the number of test images
{
	float * projectedTestFace = 0;
	int i,offset=0,trainoffset=0;
	CvRect *face=0;

	// project the test images onto the PCA subspace
	projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );
	int *idnum;
	idnum = (int*)cvAlloc(nWebCam*sizeof(int) );

	for(i=0; i<nTestFaces; i++)
	{
		int iNearest=-1, truth;
		int nearest;
		double t5=(double)cvGetTickCount();

		if(sel==1)
		{
			//face detection and eye detection 
			faceImgArr[i]=cvCreateImage(cvSize(112,96),IPL_DEPTH_8U,1);
			face=face_detect(pFaceImg[i],faceImgArr[i]);	
			cvWaitKey(0);

			if(!face)
			{ 
				printf("can not recognize!!\n"
				"face detection  failed\n\n");
				offset=i*sizeof(personName->data.s)+offset;
				continue;
			} 
		}

		// project the test image onto the PCA subspace			
		cvEigenDecomposite(
		faceImgArr[i],
		nEigens,
		eigenVectArr,
		0, 0,
		pAvgTrainImg,
		projectedTestFace);

		iNearest = findNearestNeighbor(trainPersonNumMat,projectedTestFace);
		idnum[i]=iNearest;
		t5=(double)cvGetTickCount()-t5;
		printf("face recognition time = %gms\n\n",t5/((double)cvGetTickFrequency()*1000.));

		if(iNearest==-1)
		{
			printf("can not recognize,no pepole\n\n");
			if(sel==1) offset=i*sizeof(personName->data.s)+offset;
			continue;
		}
		else
		{
			//train data
			//trainoffset=addsum(iNearest)*sizeof(trainpersonName->data);
			//nearest  = trainPersonNumMat->data.i[iNearest];
			printf("\trecognition result:\n\tID number= %d\n\n", iNearest);

			if(sel==1)
			{ 
				//test data
				truth    = personNumTruthMat->data.i[i];
				offset=i*sizeof(personName->data.s)+offset;
				printf("\tGround truth:\n\tID number= %d\n\tID name=%s \n", truth,&personName->data.s[offset]);
			}

			//show select face region
			/*cvNamedWindow("normalized face Image result",1);	
			cvShowImage("normalized face Image result",faceImgArr[i]);
			cvWaitKey(0);         
			cvDestroyWindow("normalized face Image result");*/
			//getch();
		}							
	} 

	if(sel!=1)decideredsult(idnum,trainPersonNumMat,trainpersonName,sel); 
	cvFree(&idnum);
}
//////////////////////////////////
// loadTrainingData()
//
int loadTrainingData(CvMat ** pTrainPersonNumMat,CvMat **ptrainpersonName,CvMat **trainaverage)
{
	CvFileStorage * fileStorage;
	int i;
  
	// create a file-storage interface
	fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_READ );
	if( !fileStorage )
	{
		fprintf(stderr, "Can't open facedata.xml\n");
		return 0;
	}

	nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	thre_alldata = cvReadIntByName(fileStorage, 0,  "threshold_alldata", 0);
	thres_eachdata= (CvMat *)cvReadByName(fileStorage, 0, "threshold_eachdata", 0);
	projectmedian = (CvMat *)cvReadByName(fileStorage, 0, "projectmedian", 0);
    *trainaverage=(CvMat *)cvReadByName(fileStorage, 0, "projectmedian_eachclass", 0);
	*pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
    *ptrainpersonName=(CvMat *)cvReadByName(fileStorage, 0, "PersonName", 0);
	eigenValMat  = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
	projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
	pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
	eigenVectArr = (IplImage **)cvAlloc(nTrainFaces*sizeof(IplImage *));
	
	for(i=0; i<nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
	}

	// release the file-storage interface
	cvReleaseFileStorage( &fileStorage );

	return 1;
}

//////////////////////////////////
// findNearestNeighbor()
//
int findNearestNeighbor(CvMat *trainPersonNumMat,float * projectedTestFace)
{
	double leastDistSq = 1e12;
 	int i, iTrain, iNearest = 0,iNearest2=0,number=0;
	
	iNearest2=recog_stranger(projectedTestFace);	

	if(iNearest2==-1)
	{
		return -1;
	}
    
	double *dist_temp;
	int *idnumber,compare[3]={0};
	dist_temp=(double*)cvAlloc(nTrainFaces*sizeof(double));
	idnumber=(int*)cvAlloc(nTrainFaces*sizeof(int));

	for(iTrain=0; iTrain<nTrainFaces; iTrain++)
	{
        double distSq=0;
 
		for(i=0; i<nEigens; i++)
		{
			float d_i =
				projectedTestFace[i] -
				projectedTrainFaceMat->data.fl[iTrain*nEigens + i];
        		distSq += d_i*d_i; // Euclidean
				
		} 
	
		dist_temp[iTrain]=sqrt(distSq);
        idnumber[iTrain]=iTrain;
	}
	
	//quick sort
	

	/*for(iTrain=0; iTrain<nTrainFaces; iTrain++)
	{
		printf("%d:%f\n",idnumber[iTrain],dist_temp[iTrain]);
	}*/
	//printf("\n\nafter quick sort:\n");

    //quick sort
	i=nTrainFaces-1;
	quick_sort(dist_temp,idnumber,0,i,nTrainFaces);
	
	/*for(iTrain=0; iTrain<nTrainFaces; iTrain++)
	{
		printf("%d:%f\n",idnumber[iTrain],dist_temp[iTrain]);
	}*/

	for(i=0;i<3;i++){
		
		int temp_com=0;
	
		temp_com=idnumber[i];
		compare[i]= trainPersonNumMat->data.i[temp_com];
		printf("%d:%f\n",compare[i],dist_temp[i]);
	}

	decideredsult(compare,0,0,1);

	cvFree(&dist_temp);
    cvFree(&idnumber);

	number=compare[0];
    number = number-1;
	iNearest2=threshold(projectedTestFace,number);	

	if(iNearest2==-1)
	{
	   return -1;
	}
	else
	{
	   return compare[0];
	}
	
}

//////////////////////////////////
// recog_stranger()
//
int recog_stranger(float * projectedTestFace)
{
	double distSq=0;
 	int i, iNearest =0;
	for(i=0; i<nEigens; i++)
	{
			float d_i =
				projectedTestFace[i] -projectmedian->data.fl[i];
			    distSq += d_i*d_i; // Euclidean
				
	}
	
	distSq=sqrt(distSq);
	
	printf("compute distance from point to median of all data %f\n",distSq);	

    if(distSq>thre_alldata)
    {
	   iNearest = -1;
	}

	return iNearest;
}
//////////////////////////////////
// threshold()
//
int threshold(float * projectedTestFace,int Nearest)
{
	//double leastDistSq = 1e12;
	double distSq=0;
 	int i,g=0;

	for(i=0; i<nEigens; i++)
	{
		float d_i =projectedTestFace[i]-projectedaverage->data.fl[Nearest*nEigens+i];
		distSq += d_i*d_i; // Euclidean				
	}
    
	distSq=sqrt(distSq);
  
	printf("compute distance from point to median of signle class %f\n",distSq);	

	if(distSq>thres_eachdata->data.fl[Nearest])
    {
	   g = -1;
	}
 
	return g;
}

int loadFaceImgArray(char * filename)
{
	FILE * imgListFile = 0;
	char imgFilename[512];
	int iFace, nFaces=0,offset=0;
  

	// open the input file
	if( !(imgListFile = fopen(filename, "r")) )
	{
		fprintf(stderr, "Can\'t open file %s\n", filename);
		return 0;
	}

	// count the number of faces
	while( fgets(imgFilename, 512, imgListFile) ) ++nFaces;
	rewind(imgListFile);

	// allocate the face-image array and person number matrix
    pFaceImg= (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
	faceImgArr= (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
	personNumTruthMat = cvCreateMat( 1, nFaces, CV_32SC1 );
	personName= cvCreateMat( 1, 100, CV_AUTO_STEP );

	// store the face images in an array
	for(iFace=0; iFace<nFaces; iFace++)
	{
		offset=iFace*sizeof(personName->data)+offset;
		
		// read person number and name of image file
		fscanf(imgListFile,
		"%d %s %s", personNumTruthMat->data.i+iFace,personName->data.s+offset, imgFilename);

		// load the face image
	    pFaceImg[iFace] = cvLoadImage((char*)imgFilename,0);
		
		if( !pFaceImg[iFace])
		{
			fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
			return 0;
		}
	
	}

	fclose(imgListFile);

	return nFaces;
}

//////////////////////////////////
// capture  ten video frame 
//
void capturevideoframe()
{
     int i,num,counter=0;
     IplImage *pVideoFrame=0,*pVideoFrameCopy=0,*Gray=0;
     CvRect *face=0;
     IplImage* tpl=0; 
	
     faceImgArr= (IplImage **)cvAlloc( nWebCam*sizeof(IplImage *) );  
     

	 for(i=0,num=0; num<nWebCam; i++)
	 {  
        	
		   faceImgArr[num]=cvCreateImage(cvSize(112,96),IPL_DEPTH_8U,1);

		   pVideoFrame=cvQueryFrame(video);
		
		   		   
		   cvShowImage("Video",pVideoFrame);
		   cvWaitKey(33);

		   if(!pVideoFrame)
		   {
				exit(0);
				fprintf(stderr,"capture frame from video failed!");
		   }
	
		   pVideoFrameCopy=cvCreateImage(cvGetSize(pVideoFrame),IPL_DEPTH_8U,3); 
        
	
					
		   if((pVideoFrame->origin)==1) // 1 means the image is inverted
		   {
				cvFlip(pVideoFrame, pVideoFrameCopy, 0 );
				pVideoFrame->origin = 0;
		   }
		   else
		   {
				cvCopy(pVideoFrame,pVideoFrameCopy, 0 );
				pVideoFrameCopy->origin =0;
		   }

		               
		   Gray=cvCreateImage(cvGetSize(pVideoFrame),IPL_DEPTH_8U,1);
		   cvCvtColor(pVideoFrameCopy,Gray, CV_BGR2GRAY );
		    
		  //  cvNamedWindow("Video",CV_WINDOW_AUTOSIZE);	
		    //cvShowImage("Video",Gray);
		   // cvWaitKey(0);

		   if(!face||!num||(counter==3))
		   {
				//face detection and eye detection 
				face=0;
				counter=0;
				face=face_detect(Gray,faceImgArr[num]);

				if(face) 
				{

					object_x0	  = face->x;
					object_y0     = face->y;	
					object_x0_ini = object_x0 ;
					object_y0_ini = object_y0 ;

					/* create template image */
					tpl = cvCreateImage( cvSize(TPL_WIDTH, TPL_HEIGHT), Gray->depth, Gray->nChannels );
					 
					/* extract template from frame */
					cvCopy( faceImgArr[num], tpl);
					num++;
					continue;
				 }
			}
			else
			{ 
					int sel=0;
					/* image for template matching result */
					IplImage* match_res = cvCreateImage( cvSize( 2 * SEARCH_RANGE_X+ 1,
							   2 * SEARCH_RANGE_X + 1 ),
					   IPL_DEPTH_32F, 1 );


					/* track object if template has been selected */
					if(tpl) 
					{
						IplImage* temp=cvCreateImage(cvSize(2*(face->width),2*(face->height)),IPL_DEPTH_8U,1);

						double t8=(double)cvGetTickCount();
						temp=track_object(Gray,tpl,match_res,face);
						t8=(double)cvGetTickCount()-t8;
						printf("\nface tracking time = %gms\n",t8/((double)cvGetTickFrequency()*1000.));	

						if(temp)sel=eyes_detect_normalize(temp,faceImgArr[num]);
						 
						if(sel)
						{
						   // cvNamedWindow("facetracking",1);	
						    //cvShowImage("facetracking",temp);
							//cvWaitKey(0);
							cvDestroyWindow("facetracking");
							num++; 
							continue;
						}
						else  counter++;
					}
		   }
	 }
		/* free memory */
	   // cvDestroyWindow("Video");
		cvReleaseImage(&pVideoFrameCopy);  
		cvReleaseImage(&Gray);
 }

//////////////////////////////////
// exit: Here all the memory releasing functions
//
void exitprogram(){
    
	//facedet
	closeFaceDet();
   
	//ptr
	cvFree(&pFaceImg);
	cvFree(&faceImgArr);
    cvFree(&eigenVectArr);

	//Image
	cvReleaseImage(&pAvgTrainImg);

	//matrix data
    cvReleaseMat(&projectedTrainFaceMat);
	cvReleaseMat(&eigenValMat);		
	
	if(personNumTruthMat)
	{
	  cvReleaseMat(&personNumTruthMat);
	  cvReleaseMat(&personName);
	}
	exit(0);
}

//////////////////////////////////
// math:summation
//
int addsum(int a){
 
	if(a==0)
	{
	 return a;
	}
	else
	{
	return a+addsum(a-1);
	}
}

void decideredsult(int *idnumber,CvMat *trainPersonNumMat,CvMat *trainpersonName,int sel)
{

   int num[100][2]; //0-ID number,1-times
   int temp,name,trainoffset; 
   int i,j; 
   
   if(sel==1){
     
		  for(i = 0;i < 3;i++)
		  {
		    temp=idnumber[i];
				
			 for(j = 0;j <3;j++)
			 {
					if(idnumber[j]==temp)
					{
					   num[j][1] ++;
					   continue;
					}
			 }
			 num[i][0] = temp;
		     num[i][1] = 1; 
		  }  

          temp = 0;
		  for(i = 0;i <3;i++)
		  {
				 if(num[temp][1] < num[i][1])
				 {
					temp = i;
				 }
		  }

		  idnumber[0]=num[temp][0];
		  return;            
   }	 
  
   else  if(sel==2)
   {
		  
		  for(i = 0;i < nWebCam;i++)
		  {
		    temp=idnumber[i];
				
			 for(j = 0;j <  nWebCam;j++)
			 {
					if(idnumber[j]==temp)
					{
					   num[j][1] ++;
					   continue;
					}
			 }
			 num[i][0] = temp;
		     num[i][1] = 1; 
		  }  

          temp = 0;
		  for(i = 0;i < nWebCam;i++)
		  {
				 if(num[temp][1] < num[i][1])
				 {
					temp = i;
				 }
		  }
             
		  name = 0;  
		  for(i = 0;i < nTrainFaces;i++)
		  {
			 if(num[temp][0] == trainPersonNumMat->data.i[i])
			 {
				name = i;
			 }
		  }	 
		  trainoffset=addsum(name)*sizeof(trainpersonName->data);
				   
		  if(num[temp][0]>0)
		  {
			  printf("\n**************************************************\n"
					"\tIn %d frame,recognition result:\n\t"
					"ID number= %d\n"
					"\n**************************************************\n",nWebCam,num[temp][0]); 
		  }
		  else
		  {
		      printf("\n**************************************************\n"
					"\tStranger\n\t"
					"Who are you?\n"
					"\n**************************************************\n");
		  }
   }
}

//////////////////////////////////
// function to track object
//
IplImage* track_object(IplImage	*frame,IplImage	*tpl,IplImage	*match_res,CvRect *face)
{
		int		win_x0, win_y0, win_x1, win_y1;
		CvPoint	minloc, maxloc;
		double	minval, maxval;
		IplImage* result=cvCreateImage(cvSize(2*(face->width),2*(face->height)),frame->depth,frame->nChannels);

		/* setup search window */
		win_x0 = object_x0 - SEARCH_RANGE_X;
		win_y0 = object_y0 - SEARCH_RANGE_Y;
		win_x1 = win_x0 + TPL_WIDTH  + ( 2 * SEARCH_RANGE_X );
		win_y1 = win_y0 + TPL_HEIGHT + ( 2 * SEARCH_RANGE_Y );

		/* make sure the window still inside the frame */
		if( win_x0 < 0 ) 
		{
			win_x0 = 0;
			win_x1 = TPL_WIDTH + ( 2 * SEARCH_RANGE_X );
		}

		if( win_y0 < 0 ) 
		{
			win_y0 = 0;
			win_y1 = TPL_HEIGHT + ( 2 * SEARCH_RANGE_Y );
		}

		if( win_x1 > frame->width ) 
		{
			win_x1 = frame->width;
			win_x0 = frame->width - TPL_WIDTH - ( 2 * SEARCH_RANGE_X );
		}

		if( win_y1 > frame->height ) 
		{
			win_y1 = frame->height;
			win_y0 = frame->height - TPL_HEIGHT - ( 2 * SEARCH_RANGE_Y );
		}
		/* apply window to frame */
		cvSetImageROI( frame, cvRect( win_x0, 
								  win_y0, 
								  win_x1 - win_x0, 
								  win_y1 - win_y0 ) );

		/* search for matching object */
		cvMatchTemplate( frame, tpl, match_res, CV_TM_SQDIFF_NORMED );
		cvMinMaxLoc( match_res, &minval, &maxval, &minloc, &maxloc, 0 );

		/* clear window */
		cvResetImageROI( frame );

		/* if object found... */
		if( minval <= THRESHOLD )
		{
			/* save object's current location */
			object_x0 = object_x0 - SEARCH_RANGE_X + minloc.x;
			object_y0 = object_y0 - SEARCH_RANGE_Y + minloc.y;
			face->x=object_x0;
			face->y=object_y0;
			        
			        

			/* and draw a box there */
			cvSetImageROI( frame,*face);
			cvResize(frame,result,CV_INTER_AREA);	 
			return result;
		}
		/* if not found... */
		else 
		{
			/* search around initialization area */
			object_x0 = object_x0_ini;
			object_y0 = object_y0_ini;
			return NULL;
		}

}
void quick_sort(double *A,int *B,int left,int right,int k)
{
    int low,upper,temp; 
    double point;

    if(left < right) 
	{ 
        point = A[left];		
        temp = B[left]; 
        low = left; 
        upper = right+1; 

        while(1) 
		{ 
            while(A[++low] < point&&(low<right)) ;    // 向右找 
            while(A[--upper] > point) ;  // 向左找 
            if(low >= upper) 
                break; 
            swap(&A[low], &A[upper]); 
			intswap(&B[low], &B[upper]); 
        } 

        A[left] = A[upper]; 
        A[upper] = point; 
	    B[left] = B[upper]; 
        B[upper] = temp; 
  
  
        quick_sort(A,B,left, upper-1,k);   // 對左邊進行遞迴 
        quick_sort(A,B,upper+1, right,k);  // 對右邊進行遞迴 
    } 		
}
void swap(double *a,double *b)
{
	double temp;
	temp=*a;
	*a=*b;
	*b=temp;
}
void intswap(int *a,int *b)
{
	int temp;
	temp=*a;
	*a=*b;
	*b=temp;
}
