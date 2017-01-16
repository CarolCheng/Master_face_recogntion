// facedet.h - by Robin Hewitt, 2007
// http://www.cognotics.com/opencv/downloads/camshift_wrapper
// This is free software. See License.txt, in the download
// package, for details.


// Public interface for face detection
int initPathDet(const char * haarCascadePath,const char * eyeCascadePath);
void     closeFaceDet();
CvRect* face_detect( IplImage* img,IplImage* imgdst2 );
CvRect* eyes_detect_normalize( IplImage* img2,IplImage* imgdst);
