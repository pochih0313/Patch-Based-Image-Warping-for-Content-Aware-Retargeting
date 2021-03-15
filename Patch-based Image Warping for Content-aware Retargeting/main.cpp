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

    // find patch significance and saliency color
    uchar *s;
    uint *seg;
    for (int i = 0; i < saliency.rows; i++) {
        s = saliency.ptr<uchar>(i);
        seg = segments.ptr<uint>(i);
        for (int j = 0; j < saliency.cols; j++) {
            for (int index = 0; index < 3; index++) {
                int group = seg[j];
                patch[group].significance_color.val[index] += s[j * 3 + index] / (double)patch[group].size;
                patch[group].saliency_value += s[j * 3 + index] / 3.0f / 255.0f / (double)patch[group].size;
            }
        }
    }

    // Find min and max saliency
    double min_saliency = 2e9;
    double max_saliency = -2e9;
    for (int patch_index = 0; patch_index < patch_num; ++patch_index) {
        min_saliency = min(min_saliency, patch[patch_index].saliency_value);
        max_saliency = max(max_saliency, patch[patch_index].saliency_value);
    }

    // Normalize
    for (int patch_index = 0; patch_index < patch_num; ++patch_index) {
        patch[patch_index].saliency_value = (patch[patch_index].saliency_value - min_saliency) / (max_saliency - min_saliency);
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

// double sigColor_to_salValue(cv::Scalar &sig_color)
// {
    
// }

void build_for_warping()
{
    // // Get saliency value of each patch
    // for (int patch_index = 0; patch_index < patch_num; patch_index++) {
    //     patch[patch_index].saliency_value = sigColor_to_salValue(patch[patch_index].significance_color);
    // }


    // Set up graph and mesh data
    unsigned int mesh_cols = (unsigned int)(segments.cols - 1 / grid_size) + 1;
    unsigned int mesh_rows = (unsigned int)(segments.rows - 1 / grid_size) + 1;
    float mesh_width = (float) (segments.cols - 1) / (mesh_cols - 1);
    float mesh_height = (float) (segments.rows - 1) / (mesh_rows - 1);
    
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

            // mesh
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
    
    // Set up edge list of each patch
    vector<unsigned int> edge_list_of_patch[patch_num];
    for (int edge_index = 0; edge_index < graph.edges.size(); edge_index++) {
        unsigned int v1 = graph.edges[edge_index].pair_indice.first;
        unsigned int v2 = graph.edges[edge_index].pair_indice.second;

        uint *seg1, *seg2;
        int vy1 = (int) graph.vertices[v1][1];
        int vy2 = (int) graph.vertices[v2][1];
        int vx1 = (int) graph.vertices[v1][0];
        int vx2 = (int) graph.vertices[v2][0];

        seg1 = segments.ptr<uint>(vy1);
        seg2 = segments.ptr<uint>(vy2);
        int patch_index1 = seg1[vx1];
        int patch_index2 = seg2[vx2];
        
        if (patch_index1 == patch_index2) {
            edge_list_of_patch[patch_index1].push_back(edge_index);
        } else {
            edge_list_of_patch[patch_index1].push_back(edge_index);
            edge_list_of_patch[patch_index2].push_back(edge_index);
        }
    }

    // Set up cplex variable
    IloEnv env;
    IloNumVarArray Vp(env);
    IloExpr D(env);

    for (unsigned int i = 0; i < graph.vertices.size(); i++) {
        Vp.add(IloNumVar(env, -IloInfinity, IloInfinity)); // x
        Vp.add(IloNumVar(env, -IloInfinity, IloInfinity)); // y
    }
    
    // Patch transformation constraint DTF
    const double alpha = 0.8f;
    const double width_ratio = (double) target_width / segments.cols - 1;
    const double height_ratio = (double) target_height / segments.rows - 1;

    for (unsigned int patch_index = 0; patch_index < patch_num; patch_index++) {
        const vector<unsigned int> edge_list = edge_list_of_patch[patch_index];

        if (!edge_list.size()) {
            continue;
        }

        // Find geometry tranformation T
        const Edge &center_edge = graph.edges[edge_list[0]]; // select the first edge as center edge

        // Find inverse matrix of C
        double c_x = graph.vertices[center_edge.pair_indice.first][0] - graph.vertices[center_edge.pair_indice.second][0];
        double c_y = graph.vertices[center_edge.pair_indice.first][1] - graph.vertices[center_edge.pair_indice.second][1];

        double matrix_a = c_x;
        double matrix_b = c_y;
        double matrix_c = c_y;
        double matrix_d = -c_x;

        double matrix_rank = matrix_a * matrix_d - matrix_b * matrix_c;
        if (fabs(matrix_rank) <= 1e-9) {
            matrix_rank = (matrix_rank > 0 ? 1 : -1) * 1e-9;
        }

        double inverse_matrix_a = matrix_d / matrix_rank;
        double inverse_matrix_b = -matrix_b / matrix_rank;
        double inverse_matrix_c = -matrix_c / matrix_rank;
        double inverse_matrix_d = matrix_a / matrix_rank;

        // cout << patch[patch_index].saliency_value << endl;

        for (unsigned int i = 0; i < edge_list.size(); i++) {
            const Edge &edge = graph.edges[edge_list[i]];
            
            double e_x = graph.vertices[edge.pair_indice.first][0] - graph.vertices[edge.pair_indice.second][0];
            double e_y = graph.vertices[edge.pair_indice.first][1] - graph.vertices[edge.pair_indice.second][1];

            double t_s = inverse_matrix_a * e_x + inverse_matrix_b * e_y;
            double t_r = inverse_matrix_c * e_x + inverse_matrix_d * e_y;

            // DST
            D += alpha * patch[patch_index].saliency_value *
                (IloPower((Vp[edge.pair_indice.first * 2] - Vp[edge.pair_indice.second * 2]) -
                        (t_s * (Vp[center_edge.pair_indice.first * 2] - Vp[center_edge.pair_indice.second * 2]) +
                        t_r * (Vp[center_edge.pair_indice.first * 2 + 1] - Vp[center_edge.pair_indice.second * 2 + 1])), 2) +
                Ilopower((Vp[edge.pair_indice.first * 2 + 1] - Vp[edge.pair_indice.second * 2 + 1]) -
                        (-t_r * (Vp[center_edge.pair_indice.first * 2] - Vp[center_edge.pair_indice.second * 2]) +
                        t_s * (Vp[center_edge.pair_indice.first * 2 + 1] - Vp[center_edge.pair_indice.second * 2 + 1])), 2));


            // DLT
            D += (1 - alpha) * (1 - patch[patch_index].saliency_value) * 
                (IloPower(Vp[edge.pair_indice.first * 2] - Vp[edge.pair_indice.second * 2] -
                          width_ratio * (t_s * (Vp[center_edge.pair_indice.first * 2] - Vp[center_edge.pair_indice.second * 2]) +
                                         t_r * (Vp[center_edge.pair_indice.fist * 2 + 1] - Vp[center_edge.pair_indice.second * 2 + 1])), 2) +
                 Ilopower(Vp[edge.pair_indice.first * 2 + 1] - Vp[edge.pair_indice.second * 2 + 1] -
                          height_ratio * (-t_r * (Vp[center_edge.pair_indice.first * 2] - Vp[center_edge.pair_indice.second * 2]) +
                                          t_s * (Vp[center_edge.pair_indice.first * 2 + 1] - Vp[center_edge.pair_indice.second * 2 + 1])), 2));
        }
    }
    
    // Grid orientation constraint DOR
    for (unsigned int edge_index = 0; edge_index < graph.edges.size(); edge_index++) {
        unsigned int v1 = graph.edges[edge_index].pair_indice.first;
        unsigned int v2 = graph.edges[edge_index].pair_indice.second;
        
        float delta_x = graph.vertices[v1][0] - graph.vertices[v2][0];
        float delta_y = graph.vertices[v1][1] - graph.vertices[v2][1];
        
        
    }
    
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

    build_for_warping();
    
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
