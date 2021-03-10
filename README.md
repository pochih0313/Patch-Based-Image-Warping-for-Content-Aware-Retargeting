# Patch-Based-Image-Warping-for-Content-Aware-Retargeting

## Environment
* Ubuntu 20.04
* OpenCV 3.4.13 https://github.com/opencv/opencv/archive/3.4.13.zip
* Opencv_contrib 3.4.13 https://github.com/opencv/opencv_contrib/archive/3.4.13.zip
* Cplex

## Installation
* OpenCV https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html
* Cplex https://www.ibm.com/support/knowledgecenter/SSSA5P_20.1.0/COS_KC_home.html

```
cd opencv-3.4.13
mkdir build
cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.13/modules ../
make -j4 //use 4 CPU
sudo make install
sudo ldconfig
pkg-config opencv --modversion
```

## Compile
```
g++ main.cpp -o output `pkg-config --cflags --libs opencv` -I/opt/ibm/ILOG/CPLEX_Studio_Community201/cplex/include -I/opt/ibm/ILOG/CPLEX_Studio_Community201/concert/include -DIL_STD -L/opt/ibm/ILOG/CPLEX_Studio_Community201/cplex/lib/x86-64_linux/static_pic -L/opt/ibm/ILOG/CPLEX_Studio_Community201/concert/lib/x86-64_linux/static_pic -lilocplex -lconcert -lcplex -lm -pthread
```

## Run
```
./output
```
