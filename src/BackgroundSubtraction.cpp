#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "AGMM.hpp"


using namespace cv;
using namespace std;


// play video frame by frame and show the result
int main(int argc, char** argv) {
    // open the video file for reading
    VideoCapture cap(argv[1]);

    // if not success, exit program
    if (cap.isOpened() == false) {
        cout << "Cannot open the video file" << endl;
        cin.get(); //wait for any key press
        return -1;
    }


    namedWindow("Base Video", WINDOW_NORMAL); //create a window
    namedWindow("Foreground", WINDOW_NORMAL); //create a window
    namedWindow("Background", WINDOW_NORMAL); //create a window

    Mat initFrame;
    cap.read(initFrame);
    AGMM agmm = AGMM(initFrame);

    while (true) {
        Mat frame;
        bool bSuccess = cap.read(frame); // read a new frame from video 

        //Breaking the while loop at the end of the video
        if (bSuccess == false) {
            cout << "Found the end of the video" << endl;
            break;
        }

        pair<Mat,Mat> result = agmm.updateBackground(frame);
        Mat foreground = result.first;
        Mat background = result.second;

        //show the frame in the created window
        imshow("Base Video", frame);
        imshow("Foreground", foreground);
        imshow("Background", background);


        //wait for for 10 ms until any key is pressed.  
        //If the 'Esc' key is pressed, break the while loop.
        //If the any other key is pressed, continue the loop 
        //If any key is not pressed withing 10 ms, continue the loop 
        if (waitKey(10) == 27) {
            cout << "Esc key is pressed by user. Stopping the video" << endl;
            break;
        }
    }

    return 0;
}