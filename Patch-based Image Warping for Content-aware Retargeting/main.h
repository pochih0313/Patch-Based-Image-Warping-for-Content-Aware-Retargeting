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

struct Patch{
    unsigned int id;
    unsigned int size;
    cv::Scalar segment_color;
    cv::Scalar significance_color;
};

cv::Mat segments;
Patch *patch;
unsigned int patch_num;

float grid_size = 50.0f;

double sigma = 1.0;
float k = 500;
int min_size = 50;

#endif /* main_h */