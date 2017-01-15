

// Public interface for face detection
int initPathDet(const char * haarCascadePath,const char * eyeCascadePath);
void     closeFaceDet();
CvRect* face_detect( IplImage* img,IplImage* imgdst2 );
int eyes_detect_normalize( IplImage* img2,IplImage* imgdst);
