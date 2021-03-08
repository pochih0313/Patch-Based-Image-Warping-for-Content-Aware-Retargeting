# Patch-Based-Image-Warping-for-Content-Aware-Retargeting

## Environment
* Ubuntu 20.04
* OpenCV 3.4.13 https://github.com/opencv/opencv/archive/3.4.13.zip
* opencv_contrib https://github.com/opencv/opencv_contrib

## Installation
https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html

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
g++ main.cpp -o output `pkg-config --cflags --libs opencv`
```

## Run
```
./output
```
