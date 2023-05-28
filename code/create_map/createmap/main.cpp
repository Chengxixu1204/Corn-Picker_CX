#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
using namespace cv;
using namespace std;
using namespace Eigen;
#include <string>
#include <strstream>
#include <fstream>
#include "map_drawer.h"
int main()
{
    string datapath="../../../data/pictures/";
    Mat img;
    img=imread(datapath+"19.jpg");
    map_drawer *mader= new map_drawer(img,"../../../data/maps/19.txt","../../../data/heap_maps_show/19.jpg");

    //mader->contrust_the_map();
    mader->update_img_with_map();
    return 0;
}

