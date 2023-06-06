#include <getopt.h>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <stdio.h>

#include "../include/AGMM.h"
#include "../include/Gaussian.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#ifdef WITH_IMGUI
#include <GLFW/glfw3.h> // Will drag system OpenGL headers
#include "../third_party/imGUI/imgui.h"
#include "../third_party/imGUI/imgui_impl_glfw.h"
#include "../third_party/imGUI/imgui_impl_opengl3.h"
#include "../third_party/imGUI/implot.h"
#endif

using namespace cv;
using namespace std;

#ifdef WITH_IMGUI
// Error callback for GLFW
static void glfw_error_callback(int error, const char *description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}
#endif

// Main code
int main(int argc, char **argv)
{   
    #ifdef WITH_OPENMP
    omp_set_num_threads(4);
    #endif

    if (argc < 2)
    {
        cout << "Usage: BackgroundSubtraction <video_path> [-r|--record] "
                "[-d|--debug]"
             << endl;
        return -1;
    }

    bool record = false;
    bool debug = false;

    map<char, function<void()>> options = {{'r', [&]()
                                            { record = true; }},
                                           {'d', [&]()
                                            {
                                                debug = true;
                                                record = false;
                                            }}};

    static struct option long_options[] = {{"record", no_argument, NULL, 'r'},
                                           {"debug", no_argument, NULL, 'd'},
                                           {NULL, 0, NULL, 0}};

    int option_index = 0;
    while ((option_index = getopt_long(argc, argv, "rd", long_options, NULL)) !=
           -1)
    {
        if (options.find(option_index) != options.end())
            options[option_index]();
    }

#ifdef WITH_IMGUI
    if (debug)
    {

        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
            return 1;

        const char *glsl_version = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

        // Create window with graphics context
        GLFWwindow *window = glfwCreateWindow(1920, 1080, "Background Subtraction",
                                              nullptr, nullptr);
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
        io.ConfigFlags |=
            ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();

        // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);

        // Our state
        AGMM agmm(argv[optind]);
        agmm.initializeModel();

        int frameCounter = 2;

        Mat foregroundMask, foregroundMaskBGR, foregroundImage, frame;

        tie(foregroundMask, foregroundImage, frame) = agmm.processNextFrame();

        if (frame.empty())
        {
            cout << "End of video" << endl;
            exit(0);
        }

        cvtColor(foregroundMask, foregroundMaskBGR, COLOR_GRAY2BGR);

        GLuint foregroundMaskBGRTextureID, foregroundImageTextureID, frameTextureID;

        auto generateTexture = [&](Mat &image, GLuint &textureID)
        {
            glGenTextures(1, &textureID);
            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                            GL_LINEAR); // Linear Filtering
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                            GL_LINEAR); // Linear Filtering
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_BGR,
                         GL_UNSIGNED_BYTE, image.data);
            glBindTexture(GL_TEXTURE_2D, 0);
        };

        generateTexture(foregroundMaskBGR, foregroundMaskBGRTextureID);
        generateTexture(foregroundImage, foregroundImageTextureID);
        generateTexture(frame, frameTextureID);

        static bool isPlaying = false; // New variable to play/pause the video
        static int pixelCoordinates[2] = {
            0, 0}; // New variable to control the pixel coordinates

        vector<double> etas =
            agmm.getPixelEtas(pixelCoordinates[0], pixelCoordinates[1]);
        vector<Gaussian> gaussians =
            agmm.getPixelGaussians(pixelCoordinates[0], pixelCoordinates[1]);

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();

            // Start the Dear ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            {
                ImGui::Begin("Debug Control Panel");

                ImGui::InputInt2(
                    "Pixel Coordinates",
                    pixelCoordinates); // New control to input the pixel coordinates
                ImGui::SameLine();
                if (ImGui::Button("Query Pixel"))
                {
                    etas = agmm.getPixelEtas(pixelCoordinates[0], pixelCoordinates[1]);
                    gaussians =
                        agmm.getPixelGaussians(pixelCoordinates[0], pixelCoordinates[1]);
                }

                if (ImGui::Button("Next Frame"))
                {
                    tie(foregroundMask, foregroundImage, frame) = agmm.processNextFrame();

                    if (frame.empty())
                    {
                        cout << "End of video" << endl;
                        exit(0);
                    }

                    cvtColor(foregroundMask, foregroundMaskBGR, COLOR_GRAY2BGR);

                    generateTexture(foregroundMaskBGR, foregroundMaskBGRTextureID);
                    generateTexture(foregroundImage, foregroundImageTextureID);
                    generateTexture(frame, frameTextureID);

                    etas = agmm.getPixelEtas(pixelCoordinates[0], pixelCoordinates[1]);
                    gaussians =
                        agmm.getPixelGaussians(pixelCoordinates[0], pixelCoordinates[1]);
                    frameCounter++;
                }

                if (ImGui::Button(isPlaying
                                      ? "Pause"
                                      : "Play")) // New control to play/pause the video
                {
                    isPlaying = !isPlaying;
                }

                // Display Frame Count
                ImGui::Text("Frame Count: %d", frameCounter);

                ImGui::End();
            }

            if (isPlaying)
            {
                tie(foregroundMask, foregroundImage, frame) = agmm.processNextFrame();

                if (frame.empty())
                {
                    cout << "End of video" << endl;
                    exit(0);
                }

                cvtColor(foregroundMask, foregroundMaskBGR, COLOR_GRAY2BGR);

                generateTexture(foregroundMaskBGR, foregroundMaskBGRTextureID);
                generateTexture(foregroundImage, foregroundImageTextureID);
                generateTexture(frame, frameTextureID);

                etas = agmm.getPixelEtas(pixelCoordinates[0], pixelCoordinates[1]);
                gaussians =
                    agmm.getPixelGaussians(pixelCoordinates[0], pixelCoordinates[1]);
                frameCounter++;
            }

            double widgetWidth = agmm.cols;
            double widgetHeight = agmm.rows;

            // Display foregroundMaskBGRTextureID, foregroundImageTextureID,
            // frameTextureID
            ImGui::Begin("Background Subtraction");
            ImGui::Image((void *)(intptr_t)foregroundMaskBGRTextureID,
                         ImVec2(widgetWidth, widgetHeight));
            ImGui::Image((void *)(intptr_t)foregroundImageTextureID,
                         ImVec2(widgetWidth, widgetHeight));
            ImGui::Image((void *)(intptr_t)frameTextureID,
                         ImVec2(widgetWidth, widgetHeight));
            ImGui::End();

            auto createPlot = [&](const char *title, const vector<double> &xData,
                                  const vector<double> &yData)
            {
                ImGui::Begin(title);
                if (ImPlot::BeginPlot("My Plot"))
                {
                    ImPlot::PlotLine("Points", xData.data(), yData.data(), xData.size());
                    ImPlot::EndPlot();
                }
                ImGui::End();
            };

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

            createPlot("Pixel Etas", x, y);

            vector<double> x2(256);
            vector<double> y2(256);

            for (int i = 0; i < 256; ++i)
            {
                x2[i] = i;
                double gmm = 0;
                for (const auto &gaussian : gaussians)
                {
                    gmm += gaussian.getWeight() *
                           ((1.0 / sqrt(2.0 * M_PI * gaussian.getVariance())) *
                            (exp(-pow(i - gaussian.getMean(), 2.0) /
                                 (2.0 * gaussian.getVariance()))));
                }
                y2[i] = gmm;
            }

            createPlot("Pixel GMM", x2, y2);

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
    else
#endif
    {
        AGMM agmm(argv[optind]);
        cout << "Recording" << endl;
        agmm.initializeModel();
        cout << "Model Initialized" << endl;

        Mat frame, foregroundMask, foregroundMaskBGR, foregroundImage,
            combinedFrame, resizedFrame;

        VideoWriter videoWriter;
        bool isVideoWriterInitialized = false;

        int frameCount = 1;
        cout << "Processing Frames" << endl;

        while (true)
        {
            tie(foregroundMask, foregroundImage, frame) = agmm.processNextFrame();

            frameCount++;

            if (frame.empty())
            {
                break;
            }

            cvtColor(foregroundMask, foregroundMaskBGR, COLOR_GRAY2BGR);
            hconcat(frame, foregroundMaskBGR, combinedFrame);
            resize(combinedFrame, resizedFrame, Size(), 0.5, 0.5, INTER_LINEAR);

            if (!isVideoWriterInitialized)
            {
                // List of preferred codecs
                vector<int> codecs = {
                    VideoWriter::fourcc('x', 'v', 'i', 'd'), // XVID MPEG-4
                    VideoWriter::fourcc('M', 'J', 'P', 'G'), // Motion JPEG
                    VideoWriter::fourcc('M', 'P', '4', '2'), // MPEG-4.2
                    VideoWriter::fourcc('D', 'I', 'V', '3'), // MPEG-4.3
                    VideoWriter::fourcc('D', 'I', 'V', 'X'), // MPEG-4
                    VideoWriter::fourcc('X', '2', '6', '4'), // H.264
                    VideoWriter::fourcc('H', 'E', 'V', 'C'), // H.265
                    -1                                       // Default
                };

                bool codecFound = false;

                // Try to initialize VideoWriter with each codec
                for (const auto &codec : codecs)
                {
                    if (videoWriter.open("output.avi", codec, 30, resizedFrame.size(),
                                         true))
                    {
                        codecFound = true;
                        break;
                    }
                }

                // If no codec is found, throw an error
                if (!codecFound)
                    throw runtime_error(
                        "Could not initialize VideoWriter with any codec");
                isVideoWriterInitialized = true;
            }

            videoWriter.write(combinedFrame);
            cout << "Frame Count: " << frameCount << endl;

            if (record)
            {
                imshow("Frame", resizedFrame);
                if (waitKey(1) == 27)
                {
                    break;
                }
            }
        }

        videoWriter.release();
        destroyAllWindows();
    }

    return 0;
}
