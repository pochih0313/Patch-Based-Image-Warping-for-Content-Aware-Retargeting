//
//  main.cpp
//  Patch-based Image Warping for Content-aware Retargeting
//
//  Created by 陳柏志 on 2021/1/30.
//

#include "main.h"

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

void build_graph()
{
    
}

cv::Mat create_significanceMap(cv::Mat saliency)
{
    cv::Mat result;
    result = cv::Mat::zeros(segments.rows, segments.cols, CV_8UC3);

    // find patch significance color
    uchar *s;
    uint *seg;
    for (int i = 0; i < saliency.rows; i++) {
        s = saliency.ptr<uchar>(i);
        seg = segments.ptr<uint>(i);
        for (int j = 0; j < saliency.cols; j++) {
            for (int index = 0; index < 3; index++) {
                int group = seg[j];
                patch[group].significance_color.val[index] += s[j * 3 + index] / (double)patch[group].size;
            }
        }
    }

    // return the result;
    uchar *r;
    for (int i = 0; i < segments.rows; i++) {
        seg = segments.ptr<uint>(i);
        r = result.ptr<uchar>(i);

        for (int j = 0; j < segments.cols; j++) {
            cv::Scalar color = patch[seg[j]].significance_color;
            r[j * 3] = (uchar) color[0];
            r[j * 3 + 1] = (uchar) color[1];
            r[j * 3 + 2] = (uchar) color[2];
        }
    }

    return result;
}

cv::Mat segmentation(cv::Mat source)
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
    
    // initialize patch data
    patch_num = nb_segs;
    patch = (Patch *) malloc(sizeof(Patch) * patch_num);

    for (int i = 0; i < patch_num; i++) {
        patch[i].id = i;
        patch[i].size = 0;
        for (int index = 0; index < 3; index++) {
            patch[i].segment_color.val[index] = 0;
            patch[i].significance_color.val[index] = 0;
        }
    }
    
    // find patch size
    uint *seg;
    for (int i = 0; i < segments.rows; i++) {
        seg = segments.ptr<uint>(i);

        for (int j = 0; j < segments.cols; j++) {
            patch[seg[j]].size += 1;
        }
    }
    
    // find patch segment color
    uchar *s;
    for (int i = 0; i < source.rows; i++) {
        s = source.ptr<uchar>(i);
        seg = segments.ptr<uint>(i);
        for (int j = 0; j < source.cols; j++) {
            for (int index = 0; index < 3; index++) {
                int group = seg[j];
                patch[group].segment_color.val[index] += s[j * 3 + index] / (double)patch[group].size;
            }
        }
    }

    // return the result
    uchar *r;
    for (int i = 0; i < segments.rows; i++) {
        seg = segments.ptr<uint>(i);
        r = result.ptr<uchar>(i);

        for (int j = 0; j < segments.cols; j++) {
            cv::Scalar color = patch[seg[j]].segment_color;
            r[j * 3] = (uchar) color[0];
            r[j * 3 + 1] = (uchar) color[1];
            r[j * 3 + 2] = (uchar) color[2];
        }
    }
    
    return result;
}

int main(int argc, const char * argv[]) {
    cv::Mat source_image, seg_image, sal_image, significance_map;
    
    source_image = cv::imread("res/origin.jpg");
    if (!source_image.data) {
        std::cerr << "Failed to load input image" << std::endl;
        return -1;
    }

    seg_image = cv::imread("res/segmentation.jpg");
    if (!seg_image.data) {
        std::cerr << "Failed to load input image" << std::endl;
        return -1;
    }
    seg_image = segmentation(seg_image);
    
    sal_image = cv::imread("res/saliency.jpg");
    if (!sal_image.data) {
        std::cerr << "Failed to load input image" << std::endl;
        return -1;
    }
    
    significance_map = create_significanceMap(sal_image);

    build_graph();




    
    cv::namedWindow("Source");
    cv::Mat show_source = source_image.clone();
    imshow("Source", show_source);
    
    cv::namedWindow("Segmentation");
    cv::Mat show_segmentation = seg_image.clone();
    imshow("Segmentation", show_segmentation);

    cv::namedWindow("Saliency");
    cv::Mat show_saliency = sal_image.clone();
    imshow("Saliency", show_saliency);
    
    cv::namedWindow("Significance Map", cv::WINDOW_FREERATIO);
    cv::Mat show_significance = significance_map.clone();
    imshow("Significance Map", show_significance);
    // cv::imwrite("result/significance.png", significance_map);
    
    while(1) {
        int key = cv::waitKey(0);
        
        if (key == 'q') {
            break;
        }
    }
    
    cv::destroyWindow("Source");
    return 0;
}
