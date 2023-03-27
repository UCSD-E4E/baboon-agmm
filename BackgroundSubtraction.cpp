#include "include/AGMM.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


// play video frame by frame and show the result
int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: BackgroundSubtraction <video_path>" << endl;
        return -1;
    }

    AGMM agmm(argv[1]);
    agmm.initializeModel(10);

    Mat frame, foregroundMask, foregroundMaskBGR, foregroundImage, combinedFrame, resizedFrame;

    VideoWriter output;
    bool isOutputVideoInitialized = false;

    while (true) {
        tie(foregroundMask, foregroundImage, frame) = agmm.processNextFrame();

        if (frame.empty()) {
            break;
        }

        cvtColor(foregroundMask, foregroundMaskBGR, COLOR_GRAY2BGR);
        hconcat(frame, foregroundMaskBGR, combinedFrame);
        resize(combinedFrame, resizedFrame, Size(), 0.5, 0.5, INTER_LINEAR);
    
        if (!isOutputVideoInitialized) {
            output = VideoWriter("output.avi", VideoWriter::fourcc('X', '2', '6', '4'), 25, resizedFrame.size());
            isOutputVideoInitialized = true;
        }

        output.write(resizedFrame);

        imshow("Frame", resizedFrame);

        if (waitKey(30) == 27) {
            break;
        }
    }

    agmm.~AGMM();
    output.release();
    return 0;
}