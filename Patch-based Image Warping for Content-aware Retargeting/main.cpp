//
//  main.cpp
//  Patch-based Image Warping for Content-aware Retargeting
//
//  Created by 陳柏志 on 2021/1/30.
//

#include "main.h"

using namespace std;

//cv::Scalar hsv_to_rgb(cv::Scalar c) {
//    cv::Mat in(1, 1, CV_32FC3);
//    cv::Mat out(1, 1, CV_32FC3);
//
//    float *p = in.ptr<float>(0);
//    p[0] = (float) c[0] * 360.0f;
//    p[1] = (float) c[1];
//    p[2] = (float) c[2];
//
//    cv::cvtColor(in, out, cv::COLOR_HSV2RGB);
//
//    cv::Scalar t;
//    cv::Vec3f p2 = out.at<cv::Vec3f>(0, 0);
//    t[0] = (int) (p2[0] * 255);
//    t[1] = (int) (p2[1] * 255);
//    t[2] = (int) (p2[2] * 255);
//
//    return t;
//}
//
//cv::Scalar color_mapping(int segment_id) {
//    double base = (double) (segment_id) * 0.618033988749895 + 0.24443434;
//
//    return hsv_to_rgb(cv::Scalar(fmod(base, 1.2), 0.95, 0.80));
//}

cv::Mat create_significanceMap(cv::Mat &segments, cv::Mat saliency)
{
    cv::Mat result;
    result = cv::Mat::zeros(segments.rows, segments.cols, CV_8UC3);

    // find the number of segments
    double min, max;
    cv::minMaxLoc(segments, &min, &max);
    int nb_segs = (int) max + 1;

    // find group size
    int group_size[nb_segs];
    uint *seg;

    for (int i = 0; i < nb_segs; i++) {
        group_size[i] = 0;
    }

    for (int i = 0; i < segments.rows; i++) {
        seg = segments.ptr<uint>(i);

        for (int j = 0; j < segments.cols; j++) {
            group_size[seg[j]] += 1;
        }
    }
    
    
    // find group color
    vector<cv::Scalar> group_color(nb_segs);
    uchar *s;
    for (int i = 0; i < saliency.rows; i++) {
        s = saliency.ptr<uchar>(i);
        seg = segments.ptr<uint>(i);
        for (int j = 0; j < saliency.cols; j++) {
            for (int index = 0; index < 3; index++) {
                int group = seg[j];
                group_color[group].val[index] += s[j * 3 + index] / (double)group_size[group];
            }
        }
    }

    uchar *r;

    for (int i = 0; i < segments.rows; i++) {
        seg = segments.ptr<uint>(i);
        r = result.ptr<uchar>(i);

        for (int j = 0; j < segments.cols; j++) {
            cv::Scalar color = group_color[seg[j]];
            r[j * 3] = (uchar) color[0];
            r[j * 3 + 1] = (uchar) color[1];
            r[j * 3 + 2] = (uchar) color[2];
        }
    }

    return result;
}

cv::Mat segmentation(cv::Mat source, cv::Mat &segments)
{
    cv::Mat result;
    
    cv::Ptr<cv::ximgproc::segmentation::GraphSegmentation> segmentator = cv::ximgproc::segmentation::createGraphSegmentation(sigma, k, min_size);
    segmentator->processImage(source, segments);
    result = cv::Mat::zeros(segments.rows, segments.cols, CV_8UC3);
    
    // find the number of segments
    double min, max;
    cv::minMaxLoc(segments, &min, &max);
    int nb_segs = (int) max + 1;
    std::cout << nb_segs << " segments" << std::endl;
    
    // find group size
    int group_size[nb_segs];
    uint *seg;
    
    for (int i = 0; i < nb_segs; i++) {
        group_size[i] = 0;
    }
    
    for (int i = 0; i < segments.rows; i++) {
        seg = segments.ptr<uint>(i);

        for (int j = 0; j < segments.cols; j++) {
            group_size[seg[j]] += 1;
//            cout << seg[j] << ":" << group_size[seg[j]] << endl;
        }
    }
    
    // find group color
    vector<cv::Scalar> group_color(nb_segs);
    uchar *s;
    for (int i = 0; i < source.rows; i++) {
        s = source.ptr<uchar>(i);
        seg = segments.ptr<uint>(i);
        for (int j = 0; j < source.cols; j++) {
            for (int index = 0; index < 3; index++) {
                int group = seg[j];
                group_color[group].val[index] += s[j * 3 + index] / (double)group_size[group];
            }
        }
    }

    uchar *r;

    for (int i = 0; i < segments.rows; i++) {
        seg = segments.ptr<uint>(i);
        r = result.ptr<uchar>(i);

        for (int j = 0; j < segments.cols; j++) {
//            cv::Scalar color = color_mapping(seg[j]);
            cv::Scalar color = group_color[seg[j]];
            r[j * 3] = (uchar) color[0];
            r[j * 3 + 1] = (uchar) color[1];
            r[j * 3 + 2] = (uchar) color[2];
        }
    }
    
    return result;
}

int main(int argc, const char * argv[]) {
    cv::Mat source_image, seg_image, segments, sal_image, significance_map;
    
    source_image = cv::imread("res/image.jpg");
    if (!source_image.data) {
        std::cerr << "Failed to load input image" << std::endl;
        return -1;
    }
    
    seg_image = segmentation(source_image, segments);
    
    sal_image = cv::imread("res/saliency.jpg");
    if (!sal_image.data) {
        std::cerr << "Failed to load input image" << std::endl;
        return -1;
    }
    
    significance_map = create_significanceMap(segments, sal_image);
    
    cv::namedWindow("Source");
    cv::Mat show_source = source_image.clone();
    imshow("Source", show_source);
    
    cv::namedWindow("Segmentation");
    cv::Mat show_segmentation = seg_image.clone();
    imshow("Segmentation", show_segmentation);
    
    cv::namedWindow("Significance Map");
    cv::Mat show_significance = significance_map.clone();
    imshow("Significance Map", show_significance);
    
    while(1) {
        int key = cv::waitKey(0);
        
        if (key == 'q') {
            break;
        }
    }
    
    cv::destroyWindow("Source");
    return 0;
}
