#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/segmentation.hpp>

#include "main.h"

using namespace std;
using namespace cv;

void warping(unsigned int target_width, unsigned int target_height)
{
    if (target_width <= 0 || target_height <= 0) {
        cout << "Wrong target image size" << endl;
        exit(-1);
    }

    // edge list of each patch
    for (int i = 0; i < patch_num; i++) {
        cout << patch[i].id << endl;
    }
}