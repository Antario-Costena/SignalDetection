# Traffic Light And Signal Detection
In this project, we present a supervised method of **detecting trafﬁc-signs** completely based on deep Convolutional Neural Networks (CNNs). We will use [Darknet](https://github.com/AlexeyAB/darknet), an open source neural network framework, and **Google Colaboratory**, a free environment that runs entirely in the cloud and provides a GPU.

## How to train and how to improve object detection
After loading the project [folder](https://drive.google.com/drive/folders/1H-IzMKJYn5LyHmEnwfml5stm814YWUBL?usp=sharing) on the drive, we have created a new Google Colaboratory session for training the neural network.
* in Colaboratory file, need to change the runtime type: from *Runtime* menu select *Change runtime* type and choose **GPU** as Hardware accelerator.

## Step 1. Configuration
In this section we will proceed to configure our Darknet network.
We will proceed to mount Google Drive on the Colab session.
```
from google.colab import drive
print("mounting DRIVE...")
drive.mount('/content/gdrive')
!ln -s /content/gdrive/My\ Drive/root_folder/my_drive
```
Now we will proceed to clone the [repository](https://github.com/AlexeyAB/darknet) , we're going to set some configuration parameters such as:
- [x] **OPENCV** to build with OpenCV;
- [x] **GPU** to build with CUDA to accelerate by using GPU;
- [x] **CUDNN** to build with cuDNN v5-v7 to accelerate training by using GPU;
- [x] **CUDNN_HALF** to speedup Detection 3x, Training 2x;

The next step is the compile.
```
!git clone https://github.com/AlexeyAB/darknet
%cd darknet
print("activating OPENCV...")
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile

print("engines CUDA...")
!/usr/local/cuda/bin/nvcc --version

print("activating GPU...")
!sed -i 's/GPU=0/GPU=1/' Makefile

print("activating CUDNN...")
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile

print("activating CUDNN_HALF...")
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile

print("making...")
!make

```
To proceed we will load the dataset in order to use it for training.

The idea is to insert in a folder called [obj](https://drive.google.com/drive/folders/1ZY3pJzgI33PpNdYZf1PQI1k_dgzRnprW) all the *images .jpg* with the relative *files.txt* and then compress the folder.
```
print("loading dataset...)
!cp /my_drive/dataset_folder/obj.zip ../

```
And now we can unzip it.
```
print("unziping dataset...")
!unzip ../obj.zip -d data/obj.zip ../

```
It is important to also load the main **yolo-obj.cfg** configuration file, which will contain information for the construction of the network, such as the size of the images, the number of classes, filters, any augmentation techniques and more.
The file is located in the folder [configuration_files](https://drive.google.com/drive/folders/1am7pzlCU4InMfw1gq9Rasv_S0si_wTuQ).









