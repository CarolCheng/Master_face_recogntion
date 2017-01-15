// eigenface.c, by Robin Hewitt, 2007
//
// Example program showing how to implement eigenface with OpenCV

// Usage:


#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cv.h"
#include "cvaux.h"
#include "highgui.h"
#include "cxcore.h"
#include "facedet.h"
#include <conio.h>

//// Global variables
IplImage ** faceImgArr        = 0;		// array of face images
IplImage ** pFaceImg          = 0;		// array of face images
CvMat    *  personNumTruthMat = 0;		// array of person numbers
CvMat    *  personName        = 0;		// array of person numbers
int nTrainFaces               = 0;		// the number of training images
int nEigens                   = 0;		// the number of eigenvalues
double thre_alldata			  = 0;      // threshold value of dataset
IplImage * pAvgTrainImg       = 0;		// the average image
IplImage ** eigenVectArr      = 0;		// eigenvectors
CvMat * eigenValMat           = 0;		// eigenvalues
CvMat * projectedTrainFaceMat = 0;		// projected training faces
CvMat *projectedaverage		  = 0;      // median  of projected training faces of each person
CvMat *projectmedian          = 0;      // median  of projected training faces of dataset
CvMat *thres_eachdata         = 0;      // threshold value of each person
const char *faceCascadePath="haarcascade_frontalface_alt2.xml";		//face-detection xml file
const char *eyeCascadePath="parojos.xml";							//eye-detection xml file

//// Function prototypes
void learn(char sel);
void doPCA();
void storeTrainingData();
void sing_class_dist(int human,int *number);
int  loadFaceImgArray(char * filename,char sel);
double computedistant(int human,int Train);
void swap(double *a,double *b);
void quick_sort(double *A,int left,int right,int k);
double finddistant(int human,int *number);
double computedistant(int human,int iTrain);
double allthreshold();
void mean_flex(CvMat * projecttemp,int num_comp);
void comp_mean();

//////////////////////////////////
// main()
//
void main()
{
    char sel;
	initPathDet(faceCascadePath,eyeCascadePath);
   
	printf("要進行下列何種操作:\nT:先抓取適當人臉影像,進行訓練 \nI:直接訓練\nE:結束\n");
    scanf("%c",&sel);

	if(sel!='T' && sel!='I'&& sel!='i'&& sel!='t') 
	{
		printf("結束");
		exit(0);
	}
	else
	{
		learn(sel);
	}
    closeFaceDet();
}


//////////////////////////////////
// learn()
//
void learn(char sel)
{
	int i, offset;
	CvRect* face=0;
	int human,*number;

	// load training data
	nTrainFaces = loadFaceImgArray("train.txt",sel);
	if( nTrainFaces < 2 )
	{
		fprintf(stderr,
		        "Need 2 or more training faces\n"
		        "Input file contains only %d\n", nTrainFaces);
		return;
	}
    
	//printf("一共要辨認幾個人:\n");
	//scanf("%d",&human);
	
	human=3;
    number= (int*)cvAlloc( human*sizeof(int) );
	thres_eachdata=cvCreateMat(1,human, CV_32FC1 );
	//tvalue= (double*)cvAlloc( human*sizeof(double) );

	//printf("請輸入他們的學號\n");
    
	for(i=0; i<human; i++)
	{
		number[i]=i+1;
		//scanf("%d\n",&number[i],);
	}

	if( (sel=='T') || (sel=='t'))
	{
		
		  for(i=0; i<nTrainFaces; i++)
		  { 
		    cvNamedWindow("faceImg",1);
            faceImgArr[i]=cvCreateImage(cvSize(112,96),IPL_DEPTH_8U,1);
	        face=face_detect(pFaceImg[i],faceImgArr[i]);
				
			while(!face){ 
					
				printf("\t%d can not recognize!!\n"
					   "\tface detection  failed\n",i+1);
            	i++;
				if(i<nTrainFaces)
				{
					faceImgArr[i]=cvCreateImage(cvSize(112,96),IPL_DEPTH_8U,1);
					face=face_detect(pFaceImg[i],faceImgArr[i]);
					if(face)  break;
				}
				else return;
			}

              cvShowImage("faceImg",faceImgArr[i]);
			  cvWaitKey(0);
              cvDestroyWindow("faceImg");
		   }
	}
	// do PCA on the training faces
	doPCA();

	// project the training images onto the PCA subspace
	projectedTrainFaceMat = cvCreateMat( nTrainFaces, nEigens, CV_32FC1 );
	offset = projectedTrainFaceMat->step / sizeof(float);
	for(i=0; i<nTrainFaces; i++)
	{
		//int offset = i * nEigens;
		cvEigenDecomposite(
			faceImgArr[i],
			nEigens,
			eigenVectArr,
			0, 0,
			pAvgTrainImg,
			//projectedTrainFaceMat->data.fl + i*nEigens);
			projectedTrainFaceMat->data.fl + i*offset);
	}
	
    double t3=(double)cvGetTickCount();
	sing_class_dist(human,number);
	thre_alldata=allthreshold();
	t3=(double)cvGetTickCount()-t3;
	printf("find max value = %gms\n",t3/((double)cvGetTickFrequency()*1000.));

	 //store the recognition data as an xml file
    storeTrainingData();
}


//////////////////////////////////
// storeTrainingData()
//
void storeTrainingData()
{
	CvFileStorage * fileStorage;
	int i;

	// create a file-storage interface
	fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_WRITE );

	// store all the data
	cvWriteInt( fileStorage, "nEigens", nEigens );
	cvWriteInt( fileStorage, "nTrainFaces", nTrainFaces );
	cvWriteReal( fileStorage, "threshold_alldata",thre_alldata);
	cvWrite( fileStorage, "threshold_eachdata",thres_eachdata, cvAttrList(0,0));
	cvWrite(fileStorage, "projectmedian", projectmedian, cvAttrList(0,0));
	cvWrite(fileStorage, "projectmedian_eachclass",projectedaverage, cvAttrList(0,0));
	cvWrite(fileStorage, "trainPersonNumMat", personNumTruthMat, cvAttrList(0,0));
    cvWrite(fileStorage, "PersonName", personName, cvAttrList(0,0));
	cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0,0));
	cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0,0));
	cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0,0));

	for(i=0; i<nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0,0));
	}

	// release the file-storage interface
	cvReleaseFileStorage( &fileStorage );
}



//////////////////////////////////
// doPCA()
//
void doPCA()
{
	int i;
	CvTermCriteria calcLimit;
	CvSize faceImgSize;

	// set the number of eigenvalues to use
	nEigens = nTrainFaces-1;

	// allocate the eigenvector images
	faceImgSize.width  = faceImgArr[0]->width;
	faceImgSize.height = faceImgArr[0]->height;
	eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);
	for(i=0; i<nEigens; i++)
		eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// allocate the eigenvalue array
    eigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );

	// allocate the averaged image
	pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// set the PCA termination criterion
	calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);

	// compute average image, eigenvalues, and eigenvectors
	cvCalcEigenObjects(
		nTrainFaces,
		(void*)faceImgArr,
		(void*)eigenVectArr,
		CV_EIGOBJ_NO_CALLBACK,
		0,
		0,
		&calcLimit,
		pAvgTrainImg,
		eigenValMat->data.fl);

	cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);
}


//////////////////////////////////
// loadFaceImgArray()
//
int loadFaceImgArray(char * filename,char sel)
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

	// allocate the face-image array,person number,person name matrix
	if(sel=='T'||sel=='t')  
	pFaceImg= (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
    faceImgArr= (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
	personNumTruthMat = cvCreateMat( 1, nFaces, CV_32SC1 );
	personName=  cvCreateMat( nFaces, nFaces, CV_32FC4 );

	// store the face images in an array
	for(iFace=0; iFace<nFaces; iFace++)
	{
		// read person number and name of image file
        offset=iFace*sizeof(personName->data)+offset;
		fscanf(imgListFile,
			"%d %s %s", personNumTruthMat->data.i+iFace,personName->data.s+offset, imgFilename);
		

		// load the face image
		if(sel=='T'||sel=='t')
		{
			
			pFaceImg[iFace] = cvLoadImage((char*)imgFilename,0);
				
			if( !pFaceImg[iFace])
			{
				fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
				return 0;
			}
	    }
		else{
		    
			faceImgArr[iFace] = cvLoadImage((char*)imgFilename,0);
				
			if( !faceImgArr[iFace])
			{
				fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
				return 0;
			}
		}
	}

	fclose(imgListFile);

	return nFaces;
}
//////////////////////////////////
// allthreshold():compute threshod of dataset in order to recognize stranger
// return value:   
//             type: double  
//             function:class max
double allthreshold()
{
	int i,k,*oringinal_index,mid,index;
	double distSq=0,*dist_ave,median,max;
    double* oringinal_ave;
 
	oringinal_index=(int*)cvAlloc(nTrainFaces*sizeof(int) );
    dist_ave=(double*)cvAlloc(nTrainFaces*sizeof(double) );
    oringinal_ave=(double*)cvAlloc(nTrainFaces*sizeof(double) );
    projectmedian = cvCreateMat(1,nEigens, CV_32FC1 );

   	//find mean of dataset   
    comp_mean();

	//compute distant from any point to mean
	for(i=0;i<nTrainFaces;i++)
	{
        distSq=0;
		for(k=0; k<nEigens; k++)
		{
     	    float d_i=projectedTrainFaceMat->data.fl[i*nEigens + k]-projectmedian->data.fl[k];
			distSq += d_i*d_i; // Euclidean
      	}
	    
		dist_ave[i]=sqrt(distSq);
	    oringinal_ave[i]=dist_ave[i];
        oringinal_index[i]=i;	
	} 
	
	//quick sort
	k=nTrainFaces-1;
	quick_sort(dist_ave,0,k,nTrainFaces);
	
	if(nTrainFaces%2==0)
	{
	   mid=(nTrainFaces/2);		  
	}
	else
	{
	   mid=((nTrainFaces+1)/2);	
	}  

	median= dist_ave[mid];
	max=dist_ave[k];
	printf("\nthe number of data:%d\n",nTrainFaces);
	printf("maximum:%f\nmean value:%f\n",dist_ave[k],median);

	//find position of original data
	for(i=0; i<=nTrainFaces;i++)
	{
		if( oringinal_ave[i]==median)
		{
		 index=i;		
		}
	}
	
	for(k=0; k<nEigens; k++)
	{ 
	  projectmedian->data.fl[k]=projectedTrainFaceMat->data.fl[index*nEigens + k];	 
	}   

	//compute distant from any point to median
	for(i=0;i<nTrainFaces;i++)
	{
		distSq=0;
		for(k=0; k<nEigens; k++)
		{
			float d_i =	projectedTrainFaceMat->data.fl[i*nEigens + k]-projectmedian->data.fl[k];
			distSq += d_i*d_i; // Euclidean	
		}	 
		dist_ave[i]=sqrt(distSq);
		oringinal_ave[i]=sqrt(distSq);
		oringinal_index[i]=i;	
	} 
	
	//quick sort
	k=nTrainFaces-1;
	quick_sort(dist_ave,0,k,nTrainFaces);
	
	for(i=0;i<nTrainFaces;i++)
	{
      printf("\n\t%f",dist_ave[i]);	
	} 
    printf("\n");

	if(nTrainFaces%2==0)
	{
	   mid=(nTrainFaces/2);		  
	}
	else
	{
	   mid=((nTrainFaces+1)/2);	
	}  
   
	k=3*(nTrainFaces-1)/4;
	median= dist_ave[mid];
	max=dist_ave[k];
	printf("\nthe number of data:%d\n",nTrainFaces);
	printf("maximum:%f\nmedian value:%f\n",dist_ave[k],median);

	printf("\nmaximum value:%f\n", max);


	//release momory
	cvFree(&dist_ave);
    cvFree(&oringinal_ave);
	return max;
}

//////////////////////////////////
// computedistant()
//
double computedistant(int human,int iTrain)
{
	int i;
    double distSq=0;

	for(i=0; i<nEigens; i++)
	{
		float d_i =	projectedTrainFaceMat->data.fl[iTrain*nEigens + i]-projectedaverage->data.fl[human*nEigens+i];
		distSq += d_i*d_i; // Euclidean
				
	}
     return distSq;
}

void quick_sort(double *A,int left,int right,int k)
{
    int low,upper; 
    double point;

    if(left < right) 
	{ 
        point = A[left]; 
        low = left; 
        upper = right+1; 

        while(1) 
		{ 
            while(A[++low] < point) ;    // 向右找 
            while(A[--upper] > point) ;  // 向左找 
            if(low >= upper) 
                break; 
            swap(&A[low], &A[upper]); 
        } 

        A[left] = A[upper]; 
        A[upper] = point; 
  
        quick_sort(A, left, upper-1,k);   // 對左邊進行遞迴 
        quick_sort(A, upper+1, right,k);  // 對右邊進行遞迴 
    } 		
}

void swap(double *a,double *b)
{
	double temp;
	temp=*a;
	*a=*b;
	*b=temp;
}
//////////////////////////////////
// comp_avg():compute distant from any point to mean
//
void comp_aver(int *clanum,int *number,double *dist_ave,double *oringinal_ave,int *oringinal_index,int human)
{
	int i,j,k=0,mtemp=0;
	double temp2=0;

	for(j=0; j<human; j++)
	{
	   	int temp1,temp;
		temp=clanum[j];
	    temp1=number[j];
		
		for(i=0,k=0;k<temp,i<nTrainFaces;i++)
	   {
 			 if((temp1==personNumTruthMat->data.i[i]))
			 {
				 temp2=computedistant(j,i);
				 dist_ave[mtemp+k]=sqrt(temp2);
				 oringinal_ave[mtemp+k]=sqrt(temp2);
				 oringinal_index[mtemp+k]=i;
				 k++;
			  } 
			
	   } 
	   mtemp+=temp;
	}
}
//////////////////////////////////
//sing_class_dist():compute threshold of each person
//
void sing_class_dist(int human,int *number)
{
	int i,j,*clanum,k,mtemp,*index,*oringinal_index;
	double distSq=0,*dist_ave,*median;
    double* oringinal_ave;
    
	oringinal_index=(int*)cvAlloc(nTrainFaces*sizeof(int) );
	index=(int*)cvAlloc( human*sizeof(int) );
	clanum=(int*)cvAlloc( human*sizeof(int) );
    dist_ave=(double*)cvAlloc(nTrainFaces*sizeof(double) );
    oringinal_ave=(double*)cvAlloc(nTrainFaces*sizeof(double) );
    projectedaverage = cvCreateMat( human, nEigens, CV_32FC1 );
	median=(double*)cvAlloc( human*sizeof(double*) );

	//count the number of each class 
	for(j=0; j<human; j++)
	{
		int temp=0,temp1;
		temp1=number[j];
		float value=0;
		   
		for(i=0; i<nTrainFaces; i++)
		{
			if(temp1== personNumTruthMat->data.i[i])
			{
			   temp++;
			} 
		 }  
		 clanum[j]=temp;
  	 }
   	
     printf("\n");
	//find mean of each class    
    for(j=0; j<human; j++)
	{
	   int temp1;
	   temp1=number[j];
	 
	   for(k=0; k<nEigens; k++)
	   {
		   double value=0.00;
	       float comp_val=0;

		   for(i=0; i<nTrainFaces; i++)
		   {
				if(temp1== personNumTruthMat->data.i[i])
				{
					value+=projectedTrainFaceMat->data.fl[i*nEigens + k];
				
				} 
		   } 
           comp_val=(float)(value/clanum[j]); 	
		   projectedaverage->data.fl[j*nEigens+k]=comp_val;	 
		}
	}

	//compute distant from any point to mean
	comp_aver(clanum,number,dist_ave,oringinal_ave,oringinal_index,human);
   
	// sort data with quick sort
	mtemp=0;
	for(j=0; j<human; j++)
	{
		int temp,left,right,mid=0;
		temp=clanum[j];
		left=mtemp;
		right=left+temp-1;


		quick_sort(dist_ave,left,right,temp);


		if(temp%2==0)
		{
		   mid=((right-left)/2);		  
		}
		else
		{
		   mid=((right-left+1)/2);	
		}  
        //
		median[j]= dist_ave[(left+mid)];
		thres_eachdata->data.db[j]=dist_ave[ right];
		printf("\nnumber:%d\nthe number of data:%d\n",number[j],temp);
		printf("maximum:%f\nmedian value:%f\n",dist_ave[right],median[j]);
		mtemp+=temp;
	}
	
	mtemp=0;
	for(j=0;j<human;j++)
	{ 
		int temp,left,right;
		temp=clanum[j];
		left=mtemp;
		right=left+temp-1;
		double med=median[j];

		for(k=left; k<=right;k++)
		{
			if( oringinal_ave[k]==med)
			{
			 index[j]=k;		
			}
		}
	    mtemp+=temp;
	   
	 }
	
	//find median of each class    
    for(j=0; j<human; j++)
	{
	   int temp1,temp,key;
	   key=index[j];
	   temp=oringinal_index[key];
	   temp1=number[j];
	 
	   for(k=0; k<nEigens; k++)
	   { 
		   projectedaverage->data.fl[j*nEigens+k]=projectedTrainFaceMat->data.fl[temp*nEigens + k];	 
	   }   
	
	}
   
	//compute distant from any point to median
    comp_aver(clanum,number,dist_ave,oringinal_ave,oringinal_index,human);

	printf("\n每一點到中間值的距離,決定臨界值\n");
	
	// quick sort
	mtemp=0;
	for(j=0; j<human; j++)
	{
		int temp,left,right,mid=0,mid75=0;
		temp=clanum[j];
		left=mtemp;
		right=left+temp-1;


		quick_sort(dist_ave,left,right,temp);
	
		for(i=mtemp;i<=right;i++)
		{
			printf("\n\t%f",dist_ave[i]);	
		} 

		if(temp%2==0)
		{
		   mid=((right-left)/2);
		   
		}
		else
		{
		   mid=((right-left+1)/2);	
		}  
       
	    k=3*(right-left)/4;
		median[j]= dist_ave[(left+mid)];
		thres_eachdata->data.fl[j]=(float)(dist_ave[left+k]);
		printf("\nnumber:%d\nthe number of data:%d\n",number[j],temp);
		printf("maximum:%f\nmedian value:%f\n",thres_eachdata->data.fl[j],median[j]);
		mtemp+=temp;
	}

	//release momory
	cvFree(&clanum);
	cvFree(&dist_ave);
	cvFree(&median);
    cvFree(&oringinal_ave);
	cvFree(&index);
}
//////////////////////////////////
//comp_mean():compute mean of dataset 
//
void comp_mean()
{
	int i,k;
	CvMat *proj_temp=0;
    CvScalar g;

    proj_temp=cvCreateMat(1,nTrainFaces,CV_32FC1);	
	
	for(k=0; k<nEigens;k++)
	{		  
	   float temp;
			  
	   for(i=0;i<nTrainFaces;i++) 
	   {	  				   
		   proj_temp->data.fl[i]=projectedTrainFaceMat->data.fl[i*nEigens + k];
	   } 
        g=cvAvg(proj_temp,0);
	    temp=(float)(g.val[0]/nTrainFaces);
		projectmedian->data.fl[k]=temp;
	}
}