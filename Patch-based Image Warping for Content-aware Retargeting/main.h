//
//  main.h
//  Patch-based Image Warping for Content-aware Retargeting
//
//  Created by 陳柏志 on 2021/2/2.
//

#ifndef main_h
#define main_h

#include <iostream>
#include <vector>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/segmentation.hpp>

#include "warping.h"

using namespace std;

// structures
typedef cv::Vec<float, 2> Vec2f;
typedef pair<float, float> Edge;

struct Patch {
    unsigned int id;
    unsigned int size;
    cv::Scalar segment_color;
    cv::Scalar significance_color;
};

struct Graph {
    vector<Vec2f> vertices;
    vector<Edge> edges;
};

struct Mesh {
    vector<Vec2f> vertices;
    vector<Edge> edges;
    vector<Vec2f> quads;
}

// data
cv::Mat segments;
unsigned int patch_num;
Patch *patch;
Graph graph;

float grid_size = 50.0f;

// segmentation arguments
double sigma = 1.0;
float k = 500;
int min_size = 50;

#endif /* main_h */