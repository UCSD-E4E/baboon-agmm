#include "include/AGMM.h"
#include "include/Gaussian.h"
#include <getopt.h>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "imGUI/imgui.h"
#include "imGUI/imgui_impl_glfw.h"
#include "imGUI/imgui_impl_opengl3.h"
#include "imGUI/implot.h"
#include <stdio.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

using namespace cv;
using namespace std;

static void glfw_error_callback(int error, const char *description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

double getProbability(double x, double mean, double variance)
{
    double probability = (1.0 / sqrt(2.0 * M_PI * variance)) * (exp(-pow(x - mean, 2.0) / (2.0 * variance)));
    return probability;
}

static tuple<GLuint, GLuint, GLuint> generateDisplay(AGMM &agmm)
{
    Mat foregroundMask, foregroundMaskBGR, foregroundImage, frame, combinedFrame;

    tie(foregroundMask, foregroundImage, frame) = agmm.processNextFrame();

    if (frame.empty())
    {
        cout << "End of video" << endl;
        exit(0);
    }

    cvtColor(foregroundMask, foregroundMaskBGR, COLOR_GRAY2BGR);

    GLuint foregroundMaskBGRTextureID;
    glGenTextures(1, &foregroundMaskBGRTextureID);
    glBindTexture(GL_TEXTURE_2D, foregroundMaskBGRTextureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // Linear Filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // Linear Filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, foregroundMaskBGR.cols, foregroundMaskBGR.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, foregroundMaskBGR.data);
    glBindTexture(GL_TEXTURE_2D, 0);

    GLuint foregroundImageTextureID;
    glGenTextures(1, &foregroundImageTextureID);
    glBindTexture(GL_TEXTURE_2D, foregroundImageTextureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // Linear Filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // Linear Filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, foregroundImage.cols, foregroundImage.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, foregroundImage.data);
    glBindTexture(GL_TEXTURE_2D, 0);

    GLuint frameTextureID;
    glGenTextures(1, &frameTextureID);
    glBindTexture(GL_TEXTURE_2D, frameTextureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // Linear Filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // Linear Filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, frame.data);
    glBindTexture(GL_TEXTURE_2D, 0);

    return make_tuple(foregroundMaskBGRTextureID, foregroundImageTextureID, frameTextureID);
}

// Main code
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cout << "Usage: BackgroundSubtraction <video_path> [-r|--record]" << endl;
        return -1;
    }

    bool record = false;
    int c;

    static struct option long_options[] = {
        {"record", no_argument, NULL, 'r'},
        {NULL, 0, NULL, 0}};

    while ((c = getopt_long(argc, argv, "r", long_options, NULL)) != -1)
    {
        switch (c)
        {
        case 'r':
            record = true;
            break;
        default:
            break;
        }
    }

    if (record)
    {
        AGMM agmm(argv[optind]);
        agmm.initializeModel();

        Mat frame, foregroundMask, foregroundMaskBGR, foregroundImage, combinedFrame, resizedFrame;

        VideoWriter videoWriter;
        bool isVideoWriterInitialized = false;

        while (true) {
            tie(foregroundMask, foregroundImage, frame) = agmm.processNextFrame();

            if (frame.empty())
            {
                break;
            }

            cvtColor(foregroundMask, foregroundMaskBGR, COLOR_GRAY2BGR);
            hconcat(frame, foregroundMaskBGR, combinedFrame);
            resize(combinedFrame, resizedFrame, Size(), 0.5, 0.5, INTER_LINEAR);


            if (!isVideoWriterInitialized)
            {
                videoWriter.open("output.avi", VideoWriter::fourcc('x', '2', '6', '4'), 25, resizedFrame.size());
                isVideoWriterInitialized = true;
            }

            videoWriter.write(resizedFrame);

            imshow("Background Subtraction", resizedFrame);

            // Do not wait for a key press
            if (waitKey(1) >= 0)
                break;
        }

        videoWriter.release();
        destroyAllWindows();
    }   
    else
    {

        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
            return 1;

        const char *glsl_version = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

        // Create window with graphics context
        GLFWwindow *window = glfwCreateWindow(1920, 1080, "Background Subtraction", nullptr, nullptr);
        if (window == nullptr)
            return 1;
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1); // Enable vsync

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();

        // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);

        // Our state
        AGMM agmm(argv[optind]);
        agmm.initializeModel();

        int frameCounter = 2;

        GLuint foregroundMaskBGRTextureID, foregroundImageTextureID, frameTextureID;
        tie(foregroundMaskBGRTextureID, foregroundImageTextureID, frameTextureID) = generateDisplay(agmm);

        static bool isPlaying = false;           // New variable to play/pause the video
        static int pixelCoordinates[2] = {0, 0}; // New variable to control the pixel coordinates

        vector<double> etas = agmm.getPixelEtas(pixelCoordinates[0], pixelCoordinates[1]);
        vector<Gaussian> gaussians = agmm.getPixelGaussians(pixelCoordinates[0], pixelCoordinates[1]);

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();

            // Start the Dear ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            {
                ImGui::Begin("Debug Control Panel");

                ImGui::InputInt2("Pixel Coordinates", pixelCoordinates); // New control to input the pixel coordinates
                ImGui::SameLine();
                if (ImGui::Button("Query Pixel"))
                {
                    etas = agmm.getPixelEtas(pixelCoordinates[0], pixelCoordinates[1]);
                    gaussians = agmm.getPixelGaussians(pixelCoordinates[0], pixelCoordinates[1]);
                }

                if (ImGui::Button("Next Frame"))
                {
                    tie(foregroundMaskBGRTextureID, foregroundImageTextureID, frameTextureID) = generateDisplay(agmm);
                    etas = agmm.getPixelEtas(pixelCoordinates[0], pixelCoordinates[1]);
                    gaussians = agmm.getPixelGaussians(pixelCoordinates[0], pixelCoordinates[1]);
                    frameCounter++;
                }

                if (ImGui::Button(isPlaying ? "Pause" : "Play")) // New control to play/pause the video
                {
                    isPlaying = !isPlaying;
                }

                // Display Frame Count
                ImGui::Text("Frame Count: %d", frameCounter);

                ImGui::End();
            }

            if (isPlaying)
            {
                tie(foregroundMaskBGRTextureID, foregroundImageTextureID, frameTextureID) = generateDisplay(agmm);
                etas = agmm.getPixelEtas(pixelCoordinates[0], pixelCoordinates[1]);
                gaussians = agmm.getPixelGaussians(pixelCoordinates[0], pixelCoordinates[1]);
                frameCounter++;
            }

            float widgetWidth = agmm.cols;
            float widgetHeight = agmm.rows;

            // Display foregroundMaskBGRTextureID, foregroundImageTextureID, frameTextureID
            ImGui::Begin("Background Subtraction");
            ImGui::Image((void *)(intptr_t)foregroundMaskBGRTextureID, ImVec2(widgetWidth, widgetHeight));
            ImGui::Image((void *)(intptr_t)foregroundImageTextureID, ImVec2(widgetWidth, widgetHeight));
            ImGui::Image((void *)(intptr_t)frameTextureID, ImVec2(widgetWidth, widgetHeight));
            ImGui::End();

            // Size of etas
            int size = etas.size();

            std::vector<double> x(size); // x-values
            std::vector<double> y(size); // y-values

            // fill x and y with some data...
            for (int i = 0; i < size; ++i)
            {
                x[i] = i;
                y[i] = etas[i];
            }

            ImGui::Begin("Pixel Etas");
            if (ImPlot::BeginPlot("My Plot"))
            {
                // Set x axis to increment by 1 and y axis to be from 1/6000 to 0.05

                ImPlot::PlotLine("Points", x.data(), y.data(), x.size());
                ImPlot::EndPlot();
            }
            ImGui::End();

            vector<double> x2(256);
            vector<double> y2(256);

            for (int i = 0; i < 256; ++i)
            {
                x2[i] = i;
                double gmm = 0;
                for (const auto &gaussian : gaussians)
                {
                    gmm += gaussian.getWeight() * getProbability(i, gaussian.getMean(), gaussian.getVariance());
                }
                y2[i] = gmm;
            }

            ImGui::Begin("Pixel GMM");
            if (ImPlot::BeginPlot("My Plot"))
            {
                // Set x axis to increment by 1 and y axis to be from 1/6000 to 0.05

                ImPlot::PlotLine("Points", x2.data(), y2.data(), x2.size());
                ImPlot::EndPlot();
            }
            ImGui::End();

            // Rendering
            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.45f, 0.55f, 0.60f, 1.00f); // Background color
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);
        }

        // Cleanup
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        ImPlot::DestroyContext();

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    return 0;
}
