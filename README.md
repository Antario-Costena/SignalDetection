# Traffic Light And Signal Detection
In this project, we present a supervised method of **detecting trafﬁc-signs** completely based on deep Convolutional Neural Networks (CNNs). We will use [Darknet](https://github.com/AlexeyAB/darknet), an open source neural network framework, and **Google Colaboratory**, a free environment that runs entirely in the cloud and provides a GPU.

## How to train and how to improve object detection
After loading the project [folder](https://drive.google.com/drive/folders/1H-IzMKJYn5LyHmEnwfml5stm814YWUBL?usp=sharing) on the drive, we have created a new Google Colaboratory session for training the neural network.
* in Colaboratory file, need to change the runtime type: from *Runtime* menu select *Change runtime* type and choose **GPU** as Hardware accelerator.

## 1 Configuration
In this section we will proceed to configure our Darknet network.
We will proceed to mount Google Drive on the Colab session
> from google.colab import drive print(**"mounting DRIVE...**") >
> drive.mount(**'/content/gdrive'**) >
> !ln -s /content/gdrive/My\ Drive/*root_folder*/my_drive >




