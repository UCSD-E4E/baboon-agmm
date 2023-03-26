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

    Mat frame;
    Mat background;
    Mat foreground;
    cap.read(frame);
    AGMM agmm = AGMM(frame);

    while (true) {
        bool bSuccess = cap.read(frame); // read a new frame from video 

        //Breaking the while loop at the end of the video
        if (bSuccess == false) {
            cout << "Found the end of the video" << endl;
            break;
        }

        tie(background,foreground) = agmm.updateBackground(frame);

        //show the frame in the created window
        imshow("Base Video", frame);
        imshow("Foreground", foreground);
        imshow("Background", background);


        //wait for for 10 ms until any key is pressed.  
        //If the 'Esc' key is pressed, break the while loop.
        //If the any other key is pressed, continue the loop 
        //If any key is not pressed withing 10 ms, continue the loop 
        if (waitKey(50) == 27) {
            cout << "Esc key is pressed by user. Stopping the video" << endl;
            break;
        }
    }

    return 0;
}