
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <sys/types.h>
#include <sys/stat.h>

struct stat info;

using namespace cv;
using namespace dnn;
using namespace std;

std::vector<std::string> classes;

// Initialize the parameters
float confThreshold = 0.1; // Confidence threshold
float maskThreshold = 0.1; // Mask threshold

vector<Scalar> colors;

void check_path(const char *pathname)
{
    if (stat(pathname, &info) != 0)
        printf("cannot access %s\n", pathname);
    else if (info.st_mode & S_IFDIR) // S_ISDIR() doesn't exist on my windows
        printf("%s is a directory\n", pathname);
    else
        printf("%s is no directory\n", pathname);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net &net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat &frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255));

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat &frame, const vector<Mat> &outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    float confThreshold = 0;
    float nmsThreshold = 0.1;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float *data = (float *)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Mat scores_2;
            scores.copyTo(scores_2);
            // for (int l = 0; l < scores.rows; l++)
            // {
            //     for (int s = 0; s < scores.cols; )
            // }
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            // printf("%d\n", classIdPoint.y);
            // printf("%f\n", scores.at<float>(0, 32));
            // printf("%f\n", scores.at<float>(classIdPoint));
            if (confidence > confThreshold)
            {
                // printf("%d\n", classIdPoint.x);
                // printf("%d\n", classIdPoint.y);
                // printf("%d\n", scores.row(classIdPoint.y).col(classIdPoint.x));
                // printf("%d\n", scores);
                cout << "M = " << endl
                     << " " << scores_2 << endl
                     << endl;
                printf("%f\n", confidence);
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                printf("%d\n", classIdPoint.x);
                printf("%s\n", classes[classIdPoint.x].c_str());
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

// Draw the predicted bounding box, colorize and show the mask on the image
void drawBox(Mat &frame, int classId, float conf, Rect box, Mat &objectMask)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        printf("Classid: %d, num_classes: %d\n", classId, (int)classes.size());
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    box.y = max(box.y, labelSize.height);
    rectangle(frame, Point(box.x, box.y - round(1.5 * labelSize.height)), Point(box.x + round(1.5 * labelSize.width), box.y + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);

    Scalar color = colors[classId % colors.size()];
    // Comment the above line and uncomment the two lines below to generate different instance colors
    //int colorInd = rand() % colors.size();
    //Scalar color = colors[colorInd];

    // Resize the mask, threshold, color and apply it on the image
    resize(objectMask, objectMask, Size(box.width, box.height));
    Mat mask = (objectMask > maskThreshold);
    Mat coloredRoi = (0.3 * color + 0.7 * frame(box));
    coloredRoi.convertTo(coloredRoi, CV_8UC3);

    // Draw the contours on the image
    vector<Mat> contours;
    Mat hierarchy;
    mask.convertTo(mask, CV_8U);
    findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    drawContours(coloredRoi, contours, -1, color, 5, LINE_8, hierarchy, 100);
    coloredRoi.copyTo(frame(box), mask);
}

// For each frame, extract the bounding box and mask for each detected object
void postprocess_mask(Mat &frame, const vector<Mat> &outs)
{
    Mat outDetections = outs[0];
    Mat outMasks = outs[1];

    // Output size of masks is NxCxHxW where
    // N - number of detected boxes
    // C - number of classes (excluding background)
    // HxW - segmentation shape
    const int numDetections = outDetections.size[2];
    const int numClasses = outMasks.size[1];

    outDetections = outDetections.reshape(1, outDetections.total() / 7);
    for (int i = 0; i < numDetections; ++i)
    {
        int classId = outDetections.at<float>(i, 1);
        if (classId == 36 || classId == 33)
        {
            // Extract the bounding box
            // int classId = static_cast<int>(outDetections.at<float>(i, 1));
            float score = outDetections.at<float>(i, 2);
            int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
            int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
            int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
            int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

            left = max(0, min(left, frame.cols - 1));
            top = max(0, min(top, frame.rows - 1));
            right = max(0, min(right, frame.cols - 1));
            bottom = max(0, min(bottom, frame.rows - 1));
            Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

            // Extract the mask for the object
            Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));
            if (classId == 36)
            {
                printf("Found ball\n");
            }
            // Draw bounding box, colorize and show the mask on the image
            drawBox(frame, classId, score, box, objectMask);
        }
    }
}

int main()
{
    std::string file = "/home/dennis/yolo/mask_coco/mscoco_label.names";
    std::ifstream ifs(file.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }

    // Load the colors
    string colorsFile = "/home/dennis/yolo/mask_coco/colors.txt";
    ifstream colorFptr(colorsFile.c_str());
    while (getline(colorFptr, line))
    {
        char *pEnd;
        double r, g, b;
        r = strtod(line.c_str(), &pEnd);
        g = strtod(pEnd, NULL);
        b = strtod(pEnd, NULL);
        colors.push_back(Scalar(r, g, b, 255.0));
    }

    float scale = 1;
    Scalar mean = 1;
    bool swapRB = true;
    const String model = "/home/dennis/yolo/yolov3.weights";
    const String config = "/home/dennis/yolo/yolov3.cfg";

    // Give the configuration and weight files for the model
    String textGraph = "/home/dennis/yolo/mask_coco/mask_rcnn_inception_v2_coco.pbtxt";
    String modelWeights = "/home/dennis/yolo/mask_coco/frozen_inference_graph.pb";
    // Runs on CPU
    // Net net = readNetFromDarknet(config, model);
    Net net = readNetFromTensorflow(modelWeights, textGraph);

    Mat frame, blob, resized;

    frame = imread("/home/dennis/yolo/test_7.jpg");
    Size size(frame.size[1], frame.size[0]);
    // resize(frame, resized, size);
    // imshow("Resized", frame);
    blobFromImage(frame, blob, 1.0, size, Scalar(), false, true);
    waitKey(0);
    // printf("%d, %d, %d, %d\n", blob.size[0], blob.size[1], blob.size[2], blob.size[3]);
    net.setInput(blob);
    std::vector<Mat> outs;
    std::vector<String> outNames(2);
    outNames[0] = "detection_out_final";
    outNames[1] = "detection_masks";
    net.forward(outs, outNames);

    postprocess_mask(frame, outs);

    // Write the frame with the detection boxes
    Mat detectedFrame;
    frame.convertTo(detectedFrame, CV_8U);
    imwrite("test_2.jpg", detectedFrame);

    // imshow("Target", frame);
    // waitKey(0);
}