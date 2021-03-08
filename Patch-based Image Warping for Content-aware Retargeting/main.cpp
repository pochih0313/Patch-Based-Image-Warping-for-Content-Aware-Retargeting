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

void build_graph_and_mesh()
{
    unsigned int mesh_cols = (segments.cols / grid_size) + 1;
    unsigned int mesh_rows = (segments.rows / grid_size) + 1;
    float mesh_width = segments.cols / (float) (mesh_cols - 1);
    float mesh_height = segments.rows / (float) (mesh_rows - 1);
    
    // graph vertices
    for (unsigned int row = 0; row < mesh_rows; row++) {
        for (unsigned int col = 0; col < mesh_cols; col++) {
            graph.vertices.push_back(Vec2f(col * mesh_width, row * mesh_height));
            // cout << graph.vertices.back() << endl;
        }
    }

    // graph edges
    for (unsigned int row = 0; row < mesh_rows - 1; row++) {
        for (unsigned int col = 0; col < mesh_cols - 1; col++) {
            unsigned int index = row * mesh_cols + col;
            unsigned int indices[4] = {index, index + mesh_cols, index + mesh_cols + 1, index + 1}; // direction: counterclockwise
            Edge edge;
            
            if (col != 0) {
                edge.pair_indice = make_pair(indices[0], indices[1]);
                graph.edges.push_back(edge);
            }
            edge.pair_indice = make_pair(indices[1], indices[2]);
            graph.edges.push_back(edge);
            edge.pair_indice = make_pair(indices[3], indices[2]);
            graph.edges.push_back(edge);
            if (row != 0) {
                edge.pair_indice = make_pair(indices[0], indices[3]);
                graph.edges.push_back(edge);
            }
            
//            if (col != 0)
//                graph.edges.push_back(Edge(make_pair(indices[0], indices[1])));
//            graph.edges.push_back(Edge(make_pair(indices[1], indices[2])));
//            graph.edges.push_back(Edge(make_pair(indices[3], indices[2])));
//            if (row != 0)
//                graph.edges.push_back(Edge(make_pair(indices[0], indices[3])));

            for (int i = 0; i < 4; i++) {
                unsigned int vertex_index = indices[i];
                mesh.vertices.push_back(Vec2f(graph.vertices[vertex_index][0], graph.vertices[vertex_index][1]));
                mesh.faces.push_back(Vec2f(graph.vertices[vertex_index][0] / (float) segments.cols, graph.vertices[vertex_index][1] / (float) segments.rows));
            }
        }
    }


//     for (int i = 0; i < graph.edges.size(); i++) {
//         cout << graph.edges[i].pair_indice.first << " " << graph.edges[i].pair_indice.second << endl;
//     }
}

void warping(unsigned int target_width, unsigned int target_height)
{
    if (target_width <= 0 || target_height <= 0) {
        cout << "Wrong target image size" << endl;
        exit(-1);
    }

    // edge list of each patch
    vector<unsigned int> edge_list_of_patch[patch_num];
    for (int edge_index = 0; edge_index < graph.edges.size(); edge_index++) {
        unsigned int v1 = graph.edges[edge_index].pair_indice.first;
        unsigned int v2 = graph.edges[edge_index].pair_indice.second;
        
        uint *seg1, *seg2;
        seg1 = segments.ptr<uint>(graph.vertices[v1][1]);
        seg2 = segments.ptr<uint>(graph.vertices[v2][1]);
        int c1 = graph.vertices[v1][0];
        int c2 = graph.vertices[v2][0];
        unsigned int patch_index1 = seg1[c1];
        unsigned int patch_index2 = seg2[c2];
        
        if (patch_index1 == patch_index2) {
            edge_list_of_patch[patch_index1].push_back(edge_index);
        } else {
            edge_list_of_patch[patch_index1].push_back(edge_index);
            edge_list_of_patch[patch_index2].push_back(edge_index);
        }
    }
    
    // Patch transformation constraint
    
    
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

    build_graph_and_mesh();
    
    unsigned int target_image_width = 200;
    unsigned int target_image_height = 200;
    warping(target_image_width, target_image_height);

    
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
