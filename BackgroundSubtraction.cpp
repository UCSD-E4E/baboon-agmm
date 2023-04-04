#include "include/AGMM.h"
#include <getopt.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// play video frame by frame and show the result
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cout << "Usage: BackgroundSubtraction <video_path> [-s|--step]" << endl;
        return -1;
    }

    bool step = false;
    int c;

    static struct option long_options[] = {
        {"step", no_argument, NULL, 's'},
        {NULL, 0, NULL, 0}};

    while ((c = getopt_long(argc, argv, "s", long_options, NULL)) != -1)
    {
        switch (c)
        {
        case 's':
            step = true;
            break;
        default:
            break;
        }
    }

    AGMM agmm(argv[optind]);
    agmm.initializeModel(10);

    Mat frame, foregroundMask, foregroundMaskBGR, foregroundImage, combinedFrame, resizedFrame;

    VideoWriter videoWriter;
    bool isVideoWriterInitialized = false;

    while (true)
    {
        tie(foregroundMask, foregroundImage, frame) = agmm.processNextFrame();

        if (frame.empty())
        {
            break;
        }

        cvtColor(foregroundMask, foregroundMaskBGR, COLOR_GRAY2BGR);
        hconcat(frame, foregroundMaskBGR, combinedFrame);
        resize(combinedFrame, resizedFrame, Size(), 0.5, 0.5, INTER_LINEAR);

        if (!isVideoWriterInitialized && !step)
        {
            videoWriter.open("output.avi", VideoWriter::fourcc('x', '2', '6', '4'), 25, resizedFrame.size());
            isVideoWriterInitialized = true;
        }

        if (!step)
        {
            videoWriter.write(resizedFrame);
        }

        imshow("Background Subtraction", resizedFrame);

        int key = waitKey(step ? 0 : 30);
        if (key == 27)
        {
            break;
        }
        else if (step && key == 32)
        {
            continue;
        }
    }

    videoWriter.release();
    return 0;
}
