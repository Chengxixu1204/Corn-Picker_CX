#ifndef MAP_DRAWER_H
#define MAP_DRAWER_H
#include <opencv2/opencv.hpp>
using namespace cv;
#include <iostream>
using namespace std;
#include <Eigen/Dense>
using namespace Eigen;
#include <vector>
#include <fstream>
#include <strstream>
class map_drawer
{
public:
    map_drawer();
    map_drawer(Mat img_in,string map_in,string imgsavepath);

    Mat img_in_,img_with_map_;
    int height_,width_;
    MatrixXf map_;
    vector<Vector2i> pointvec;
    string map_in_;
    string img_save_path_;


    Vector3i color(float v);

    void update_img_with_map();
    void update_the_map(Vector2i p1,Vector2i p2);
    void contrust_the_map();
};

#endif // MAP_DRAWER_H
